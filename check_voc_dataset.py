import os
import json
from lxml import etree
from PIL import Image


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# 根据标签数据，生成classes.json文件
def create_voc_label_classes_json(path: str, save_path: str) -> dict:
    """
    :param path:eg:xxx/xxx/VOCdevkit/VOC2007
    :param save_path:save a classes.json
    :return:label classes dict
    """
    if isinstance(path, str) is False:
        raise Exception('path type is error')
    assert os.path.exists(path) is True, f'path:{path} is error'
    anno_path = os.path.join(path, 'Annotations')
    assert os.path.exists(anno_path) is True, f'path:{anno_path} is error'

    classes_set = set()
    for anno in os.listdir(anno_path):
        anno = os.path.join(anno_path, anno)
        with open(anno) as f:
            xml_str = f.read()

        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)['annotation']

        for obj in data['object']:
            classes_set.add(obj['name'])

    classes_dict = dict()
    for i, item in enumerate(classes_set, start=1):
        classes_dict[item] = i

    with open(save_path, 'w') as f:
        f.write(json.dumps(classes_dict, indent=4, ensure_ascii=False))

    return classes_dict


# 检查标签和图片对应关系
def check_voc_xml_image(path: str):
    """
    :param path: xxx/xxx/VOCdevkit/VOC2007
    """

    assert os.path.exists(path) is True, f'{path} is error'
    anno_file = os.path.join(path, 'Annotations')
    image_file = os.path.join(path, 'JPEGImages')
    assert os.path.exists(anno_file) is True, f'{anno_file} is error'
    assert os.path.exists(image_file) is True, f'{image_file} is error'

    for xml in os.listdir(anno_file):
        xml_path = os.path.join(anno_file, xml)
        image = xml_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
        assert os.path.exists(image) is True, f'{image} is error'
    print('Xml file and image all matched')

    for xml in os.listdir(anno_file):
        xml_path = os.path.join(anno_file, xml)
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)['annotation']
        file_name = data['filename']
        img_path = os.path.join(image_file, file_name)
        assert os.path.exists(img_path) is True, f'xml:{xml_path},img:{img_path}'

    print('From xml get the img path not error')


def check_image_mode(path: str, mode: str = 'RGB', convert: bool = False):
    """
    :param convert: Gray to RGB
    :param mode: RGB or BGR
    :param path: xxx/xxx/VOCdevkit/VOC2007
    """
    assert os.path.exists(path) is True, f'{path} is error'
    image_file = os.path.join(path, 'JPEGImages')
    assert os.path.exists(image_file) is True, f'{image_file} is error'

    for image in os.listdir(image_file):
        image_path = os.path.join(image_file, image)
        img = Image.open(image_path)
        if img.mode != mode:
            print(f'image path:{image_path},image mode:{img.mode}')
        if convert:
            img.convert(mode)
            img.save(image_path, quality=95)


if __name__ == '__main__':
    voc_file = '/home/cv/AI_Data/VOCdevkit/VOC2007'
    # voc_file = '/home/cv/AI_Data/HardHatWorker_voc/VOC2007'
    save_file = 'my_voc_classes.json'

    # classes_dict = create_voc_label_classes_json(voc_file, save_file)
    # print(f'classes_dict:{classes_dict}')
    check_voc_xml_image(voc_file)

    # check_image_mode(voc_file)
