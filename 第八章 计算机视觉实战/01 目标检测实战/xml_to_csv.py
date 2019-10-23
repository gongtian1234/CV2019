import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
 
 
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):    # 匹配后缀为xml的文件
        print(xml_file)                            # 
        tree = ET.parse(xml_file)                  # 加载为et类文件
        root = tree.getroot()                      # 获取根节点
        for member in root.findall('object'):      # 用for循环添加所有的人的位置
            try:
                value = (root.find('filename').text,       # 获取文件中存储的文件名
                         int(root.find('size')[0].text),   # 获取图片的宽
                         int(root.find('size')[1].text),   # 获取图片的高
                         member[0].text,                   # 
                         int(member[4][0].text),           # 依次为'xmin', 'ymin', 'xmax', 'ymax'
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
            except ValueError:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][1][0].text),
                         int(member[4][1][1].text),
                         int(member[4][1][2].text),
                         int(member[4][1][3].text)
                         )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
 
 
def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('labels.csv', index=None)
    print('Successfully converted xml to csv.')
 
 
main()
