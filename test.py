import xml.etree.ElementTree as ET
root = ET.parse(
    '/home/thanh/Desktop/ky5/xulyanh/BTL-XLA/data_test/48866.xml').getroot()
for type_tag in root.findall('object/bndbox'):
    value = type_tag.find("xmin")
    print(value.text)
# print(root)