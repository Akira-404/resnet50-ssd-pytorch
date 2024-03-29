import PIL.Image
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    def __init__(self, voc_root: str,
                 year: str = '2007',
                 transforms=None,
                 train_set: str = 'train.txt'):
        assert year in ['2007', '2012'], 'year must be in ["2007","2012"]'

        self.root = os.path.join(voc_root, 'VOCdevkit', f'VOC{year}')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        # train.txt path
        txt_list = os.path.join(self.root, 'ImageSets', 'Main', train_set)

        # read train.txt content and get the xml file path
        with open(txt_list, 'r') as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                             for line in read.readlines() if len(line.strip()) > 0]

        json_file = 'my_voc_classes.json'
        assert os.path.exists(json_file) is True, 'json_file is not exist'
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)  # class_dict={class1:1,class2:2}

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx: int):
        xml_path = self.xml_list[idx]
        with open(xml_path, 'r') as f:
            xml_str = f.read()

        xml = etree.fromstring(xml_str)  # Returns the root node
        data = self.parse_xml_to_dict(xml)
        annotation = data['annotation']
        img_height = int(annotation['size']['height'])
        img_width = int(annotation['size']['width'])
        height_width = [img_height, img_width]

        img_path = xml_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
        image = Image.open(img_path)

        assert 'object' in annotation, '{} lack of object informations'.format(xml_path)
        boxes = []
        labels = []
        iscrowd = []
        for obj in annotation['object']:
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj['bndbox']['xmin']) / img_width
            xmax = float(obj['bndbox']['xmax']) / img_width
            ymin = float(obj['bndbox']['ymin']) / img_height
            ymax = float(obj['bndbox']['ymax']) / img_height

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])

            if 'difficult' in obj:
                iscrowd.append(int(obj['difficult']))
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # image area

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target[height_width] = iscrowd
        target['height_width'] = height_width
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]

        with open(xml_path, 'r') as f:
            xml_str = f.read()

        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)
        annotation = data['annotation']
        height = int(annotation['size']['height'])
        width = int(annotation['size']['width'])
        return height, width

    def parse_xml_to_dict(self, xml) -> dict:
        """
        docs:
            将xml文件解析成字典形式
            <name>bob<name>
            xml.tag:node name eg:<name>
            xml.text:node content eg:bob
        :param xml:
        :return:{node.tag:node.content} eg{'size':{'w':'1','h':'2','d':'3'}}
        """
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx: int) -> dict:
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间
        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj["bndbox"]["xmin"]) / data_width
            xmax = float(obj["bndbox"]["xmax"]) / data_width
            ymin = float(obj["bndbox"]["ymin"]) / data_height
            ymax = float(obj["bndbox"]["ymax"]) / data_height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes,
                  "labels": labels,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd,
                  "height_width": height_width}

        return target

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        # images = torch.stack(images, dim=0)
        #
        # boxes = []
        # labels = []
        # img_id = []
        # for t in targets:
        #     boxes.append(t['boxes'])
        #     labels.append(t['labels'])
        #     img_id.append(t["image_id"])
        # targets = {"boxes": torch.stack(boxes, dim=0),
        #            "labels": torch.stack(labels, dim=0),
        #            "image_id": torch.as_tensor(img_id)}

        return images, targets


if __name__ == '__main__':
    import transforms
    from PIL import Image
    import random
    import matplotlib.pyplot as plt
    from draw_box_utils import draw_box
    import torchvision.transforms as ts

    category_index = {}
    try:
        json_file = open('./my_voc_classes.json', 'r')
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    # data_transform = {
    #     "train": transforms.Compose([transforms.SSDCropping(),
    #                                  transforms.Resize(),
    #                                  transforms.ColorJitter(),
    #                                  transforms.ToTensor(),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.Normalization(),
    #                                  transforms.AssignGTtoDefaultBox()]),
    #     "val": transforms.Compose([transforms.Resize(),
    #                                transforms.ToTensor(),
    #                                transforms.Normalization()])
    # }
    train_dataset = VOCDataSet(
        '/home/cv/AI_Data/hat_worker_voc/',
        '2007',
        data_transform['train'],
        train_set='train.txt')

    print(len(train_dataset))
    for index in random.sample(range(0, len(train_dataset)), k=5):
        img, target = train_dataset[index]
        img = ts.ToPILImage()(img)
        draw_box(img,
                 target["boxes"].numpy(),
                 target["labels"].numpy(),
                 [1 for i in range(len(target["labels"].numpy()))],
                 category_index,
                 thresh=0.5,
                 line_thickness=5)
        plt.imshow(img)
        plt.show()

# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./my_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
