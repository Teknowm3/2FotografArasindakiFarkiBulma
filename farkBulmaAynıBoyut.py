import cv2

def find_and_mark_difference(image1_path, image2_path, output_path):
    # Upload Image
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # İki resmin boyutlarını kontrol Check 2 images' sizes
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape

    # Check 2 images' size and make it harmonize
    if height1 != height2 or width1 != width2:
        print("Warning: Images have different dimensions. Differences will be shown without resizing.")
    else:
        print("Images have the same dimensions.")

    # Find 2 images' difference
    diff_image = cv2.absdiff(image1, image2)
    gray_diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    # Set a threshold value (for example, 30)
    threshold = 30
    _, threshold_diff = cv2.threshold(gray_diff_image, threshold, 255, cv2.THRESH_BINARY)

    # Find the contours of difference
    contours, _ = cv2.findContours(threshold_diff.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mark contours
    marked_image = image2.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show differences
    cv2.imshow("Marked Differences", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save marked image
    cv2.imwrite(output_path, marked_image)

if __name__ == "__main__":
    image1_path = "1.png" # First Image
    image2_path = "2.png" # Second Image
    output_path = "difference_output_marked.jpg"

    find_and_mark_difference(image1_path, image2_path, output_path)
