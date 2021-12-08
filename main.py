from util import *


def table_detection(img_path, show_img = False):
    img = cv2.imread(img_path)
    xml_path = img_path.replace(".png", ".xml")
    img = preprocess(img, 2)
    binimg = binarilize(img)
    if show_img:
        showImage(binimg)
    borderless = remove_border(binimg, )
    
    if show_img:
        showImage(borderless)
    vertical = horizontal = borderless.copy()

    img_height, img_width = horizontal.shape
    SCALE = 10

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(img_width / SCALE), 1))
    hor_dilate = cv2.dilate(horizontal, horizontal_kernel, iterations=20)
    _, hor_dilate = cv2.threshold(hor_dilate, 127, 255, cv2.THRESH_BINARY)
    # show image
    if show_img:
        showImage(hor_dilate)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(img_width / SCALE)))
    ver_dilate = cv2.dilate(vertical, vertical_kernel, iterations=20)
    _, ver_dilate = cv2.threshold(ver_dilate, 127, 255, cv2.THRESH_BINARY)
    # show image
    # showImage(ver_dilate)
    if show_img:
        showImage(ver_dilate)
    bw_and = cv2.bitwise_and(hor_dilate, ver_dilate)
    
    if show_img:
        showImage(bw_and)
    contours, hierarchy = cv2.findContours(
        bw_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    # len(contours)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    test = binimg.copy()


    for box in bounding_boxes:
        cv2.rectangle(test, (box[0], box[1]),
                    (box[0]+box[2], box[1]+box[3]), 255, 2)
    
    if show_img:
        showImage(test)

    cells = [c for c in bounding_boxes]


    rows = []
    while cells:
        first = cells[0]
        rest = cells[1:]
        cells_in_same_row = sorted(
            [c for c in rest if cell_in_same_row(c, first)], key=lambda c: c[0])

        row_cells = sorted([first] + cells_in_same_row, key=lambda c: c[0])
        row_cells = reduce_col(row_cells,10)
        rows.append(row_cells)
        cells = [c for c in rest if not cell_in_same_row(c, first)]

        rows.sort(key=avg_height_of_center)
        new_rows = []
    print(rows[1])
    for row in rows:
        new_row_cell = []
        for cell in row:
            cx, cy, cw, ch = cell
            cell_thresh = borderless[cy:cy+ch, cx:cx+cw]

            ver = hor = cell_thresh.copy()
            ih, iw = hor.shape


            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.dilate(cell_thresh, kernel)
            contours, hierarchy = cv2.findContours(
                opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            bounding_rects = [cv2.boundingRect(c) for c in contours]
            # hyperparameter tuning here
            MIN_CHAR_HEIGTH = 4
            MIN_CHAR_WIDTH = 2
            char_sized_bounding_rects = [
                (x, y, w, h) for x, y, w, h in bounding_rects if w >= MIN_CHAR_WIDTH and h >= MIN_CHAR_HEIGTH]
            FIXED_PX = 1
            # if char_sized_bounding_rects:
            minx, miny, maxx, maxy = math.inf, math.inf, 0, 0
            for x, y, w, h in char_sized_bounding_rects:
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x + w)
                maxy = max(maxy, y + h)
            x, y, w, h = minx, miny, maxx - minx, maxy - miny
            cropped = (cx+x-FIXED_PX, cy+y-FIXED_PX,
                    min(iw + FIXED_PX*2, w), min(ih + FIXED_PX*2, h))
            new_row_cell.append(cropped)
        # if new_row_cell != []:
        print(len(new_row_cell))
        new_rows.append(new_row_cell)
        test = binimg.copy()
    print("=================")
    bnd_rects = ravel(new_rows)
    for box in bnd_rects:
        cv2.rectangle(test, (box[0], box[1]),
                    (box[0]+box[2], box[1]+box[3]), 255, 2)

    if show_img:
        showImage(test)
    descartes = []

    # read file 
    annot_data = parseXml(xml_path)
    cell_count = len(list(annot_data))
    cell_count

    print(cell_count)
    print(len(bnd_rects))
    print("=============")
    for rect in bnd_rects:
        for data in annot_data:
            iou = calculate_iou(rect, data)
            
            descartes.append(iou)

    descartes = np.array(descartes)
    IOU = np.sort(descartes)[::-1][:cell_count]
    average = sum(IOU) / cell_count
    print(average)
    good_estimated = sum(IOU > 0.5)
    return good_estimated


if __name__ == "__main__":
    {table_detection('./data_test/' + i) for i in os.listdir('./data_test/') if i.endswith('.png')
    or i.endswith('.PNG') or i.endswith('.jpg') or i.endswith('.JPG')}
