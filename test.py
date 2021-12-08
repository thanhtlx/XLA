from util import * 

img = cv2.imread("/home/thanh/Desktop/ky5/xulyanh/BTL-XLA/data_test/48866.png")
res_data = parseXml(
    "/home/thanh/Desktop/ky5/xulyanh/BTL-XLA/data_test/48866.xml")
res_data = list(res_data)


showImage(img)

for row in res_data:
    for box in row:
        cv2.rectangle(img, (box[0], box[2]),
                      (box[1], box[3]), 255, 2)
showImage(img)
