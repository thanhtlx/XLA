from util import *


MIN_CHAR_HEIGTH = 10
MIN_CHAR_WIDTH = 10
SCALE = 100
MIN_COLUMN_SPACE = 15
INTERATIONS = 20
THRESHOLDING_TYPE = cv2.THRESH_BINARY

def table_detection(img_path, show_img = False):
    # read image 
    img = cv2.imread(img_path)
    xml_path = img_path.replace(".png", ".xml")
    
    # tien xu ly image 
    img = preprocess(img, 2)
    binimg = binarilize(img)
    borderless = remove_border(binimg)
    if show_img:
        showImage(borderless)

    # extract vertical and horizontal and merger them
    vertical = horizontal = borderless.copy()
    _, img_width = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(img_width / SCALE), 1))
    hor_dilate = cv2.dilate(
        horizontal, horizontal_kernel, iterations=INTERATIONS)
    _, hor_dilate = cv2.threshold(hor_dilate, 127, 255, THRESHOLDING_TYPE)
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(img_width / SCALE)))
    ver_dilate = cv2.dilate(vertical, vertical_kernel, iterations=INTERATIONS)
    _, ver_dilate = cv2.threshold(ver_dilate, 127, 255, THRESHOLDING_TYPE)
    # merge
    bw_and = cv2.bitwise_and(hor_dilate, ver_dilate)
    if show_img:
        showImage(bw_and)

    # find hinh chu nhat 
    contours,_ = cv2.findContours(
        bw_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    

    # merge cell both in row  
    cells = [c for c in bounding_boxes]
    rows = []
    while cells:
        first = cells[0]
        rest = cells[1:]
        cells_in_same_row = sorted(
            [c for c in rest if isInSameRow(c, first)], key=lambda c: c[0])
        row_cells = sorted([first] + cells_in_same_row, key=lambda c: c[0])
        row_cells = reduce_col(row_cells, MIN_COLUMN_SPACE)
        rows.append(row_cells)
        cells = [c for c in rest if not isInSameRow(c, first)]
        rows.sort(key=average)
    
    # remove cell invaild
    predicts = []
    for row in rows:
        for cell in row:
            cx, cy, cw, ch = cell
            cell_thresh = borderless[cy:cy+ch, cx:cx+cw]
            hor = cell_thresh.copy()
            ih, iw = hor.shape
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.dilate(cell_thresh, kernel)
            contours, _ = cv2.findContours(
                opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            bounding_rects = [cv2.boundingRect(c) for c in contours]
            bounding_vaild = [
                (x, y, w, h) for x, y, w, h in bounding_rects if w >= MIN_CHAR_WIDTH and h >= MIN_CHAR_HEIGTH]
            FIXED_PX = 2
            if bounding_vaild:
                minx, miny, maxx, maxy = math.inf, math.inf, 0, 0
                for x, y, w, h in bounding_vaild:
                    minx = min(minx, x)
                    miny = min(miny, y)
                    maxx = max(maxx, x + w)
                    maxy = max(maxy, y + h)
                x, y, w, h = minx, miny, maxx - minx, maxy - miny
                cropped = (cx+x-FIXED_PX, cy+y-FIXED_PX,
                        min(iw + FIXED_PX*2, w), min(ih + FIXED_PX*2, h))
                predicts.append(cropped)

    # drawOnImage(predicts,img)
    # cal iou index
    descartes = []
    true_data = list(parseXml(xml_path))
    for pred in predicts:
        tmp = 0
        for tr in true_data:
            iou = calculate_iou(pred,tr)
            if iou > tmp:
                tmp = iou
        descartes.append(tmp)
    if len(descartes) == 0:
        return 0
    descartes = np.array(descartes)
    res =  sum(descartes > 0.5) / descartes.shape[0]
    return res


def test():
    res = [table_detection('./data_test/' + i) for i in os.listdir('./data_test/') if i.endswith('.png')]
    res = np.array(res)
    print(sum(res > 0.5) / res.shape[0])

def train():
    res = [table_detection('./data_train/' + i) for i in os.listdir('./data_train/') if i.endswith('.png')]
    res = np.array(res)
    print(sum(res > 0.5) / res.shape[0])

if __name__ == "__main__":
    test()
    train()

