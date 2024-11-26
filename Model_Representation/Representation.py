from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_image(image_path):

    # Load the best model
    model = YOLO("runs/detect/trainXX/weights/best.pt") # Adjust trainXX
    results = model("path/to/image.jpg")

    """
    Parameters:
        image_path
    Returns:
        image: image with boxes drawn on it
    """
    image = Image.open(image_path)

    # Run the inference and retrieve the boxes
    results = model(image)
    predictions = results[0].boxes.data.tolist()

    label_names = ["ID CARD", "PASSPORT"] # Change if needed
    colors = ["orange", "blue"] # Change if needed

    draw = ImageDraw.Draw(image)
    font_path = "SpaceMono-Regular.ttf" # Change if needed
    font = ImageFont.truetype(font=font_path, size=24)

    for prediction in predictions:
        x1, y1, x2, y2, confidence, label = prediction
        label = int(label)
        # Draw the box
        draw.rectangle([(x1, y1), (x2, y2)], outline=colors[label], width=5)

        # Draw the text with the label name and confidence
        text = f"{label_names[label]} ({confidence:.3f})"
        text_width, text_height = font.getsize(text)
        text_x = x1 + 5
        text_y = y1 + 5
        draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill=colors[label])
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    return image

image_with_boxes = draw_boxes_on_image("path/to/image.jpg")
image_with_boxes.show()