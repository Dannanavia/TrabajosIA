import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Lista para almacenar los puntos del polígono
points = []


def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        # Dibujar el contorno del polígono
        if len(points) > 1:
            cv2.polylines(param, [np.array(points)], isClosed=False, color=(
                0, 255, 255), thickness=2)
        cv2.imshow("Seleccione los puntos del polígono", param)


def load_image(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
    image = cv2.imread(filepath)
    if image is None:
        raise IOError(f"No se pudo leer el archivo {
                      filepath}. Verifique la ruta y el archivo.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def segment_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Ajustar estos valores para el color deseado
    lower_bound = np.array([35, 50, 50])
    upper_bound = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask


def find_contours(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(image, contours):
    contour_image = image.copy()
    for i, contour in enumerate(contours):
        cv2.drawContours(contour_image, [contour], -1, (0, 0, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(contour_image, f"ID {
                        i + 1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return contour_image


def calculate_areas(contours, conversion_factor):
    # Convertir el área a cm²
    areas = [cv2.contourArea(contour) *
             conversion_factor for contour in contours]
    return areas


def select_polygon(image):
    global points
    points = []
    # Redimensionar la imagen para visualizarla completa
    screen_res = 1280, 690
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    resized_image = cv2.resize(image, (window_width, window_height))
    temp_image = resized_image.copy()
    cv2.imshow("Seleccione los puntos del polígono", temp_image)
    cv2.setMouseCallback("Seleccione los puntos del polígono",
                         click_event, param=temp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ajustar los puntos seleccionados a las dimensiones originales de la imagen
    adjusted_points = [(int(x / scale), int(y / scale)) for (x, y) in points]
    return adjusted_points


def process_image(filepath, pixels_per_cm, height_cm):
    image = load_image(filepath)

    # Seleccionar el polígono
    points = select_polygon(image)
    if len(points) < 3:
        raise ValueError(
            "Se requieren al menos tres puntos para formar un polígono.")

    # Calcular la altura en píxeles del polígono seleccionado
    poly_height_px = max([y for x, y in points]) - min([y for x, y in points])

    # Factor de conversión de píxeles a cm² basado en la altura del polígono
    conversion_factor = (height_cm / poly_height_px) ** 2

    # Crear la máscara basada en el polígono
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)

    # Aplicar segmentación de color solo en la región del polígono
    color_mask = segment_color(image)
    roi_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)

    contours = find_contours(roi_mask)
    contour_image = draw_contours(image, contours)
    areas = calculate_areas(contours, conversion_factor)

    # Dibujar el contorno del polígono seleccionado en amarillo
    for i in range(len(points)):
        cv2.line(contour_image, points[i], points[(
            i + 1) % len(points)], (255, 255, 0), 2)

    # Mostrar imágenes
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(image)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')

    axes[1].imshow(contour_image)
    axes[1].set_title('Imagen con Contornos')
    axes[1].axis('off')

    plt.show()

    # Crear la carpeta "Contornos" en la ruta de la imagen si no existe
    output_dir = os.path.join(os.path.dirname(filepath), "Contornos")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar imagen con contornos
    output_image_path = os.path.join(output_dir, os.path.basename(
        filepath).replace('.JPG', '_contours.JPG'))
    cv2.imwrite(output_image_path, cv2.cvtColor(
        contour_image, cv2.COLOR_RGB2BGR))

    # Guardar áreas en un archivo
    output_data_path = os.path.join(output_dir, os.path.basename(
        filepath).replace('.JPG', '_areas.txt'))
    total_area = sum(areas)
    with open(output_data_path, 'w') as f:
        for i, area in enumerate(areas):
            f.write(f'Polígono {i + 1}: {area:.2f} cm²\n')
        f.write(f'Suma total de las áreas = {total_area:.2f} cm²\n')

    return areas


# Solicitar al usuario la escala y la altura real
cm_defined = float(input("Ingrese la cantidad de centímetros definida: "))
pixels_per_cm = float(
    input(f"Ingrese la cantidad de píxeles que equivalen a {cm_defined} cm: "))
height_cm = float(input("Ingrese la altura real en cm: "))

# Archivos de imágenes
image_files = [
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_1968.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2129.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_1994.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2130.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2131.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2132.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2135.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2136.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2137.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2138.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2448.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2449.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2980.JPG',
    r'C:\Users\Oscar\Downloads\Calicatas\Imagenes\IMG_2988.JPG'
]

# Procesar cada imagen y calcular áreas
results = {}
for filepath in image_files:
    try:
        areas = process_image(filepath, pixels_per_cm, height_cm)
        results[filepath] = areas
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")

results
