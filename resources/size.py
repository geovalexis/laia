from moviepy.editor import VideoFileClip

def resize_video(input_path, output_path, width, height):
    # Cargar el video
    clip = VideoFileClip(input_path)
    
    # Cambiar el tamaño del video
    resized_clip = clip.resize(newsize=(width, height))
    
    # Guardar el video redimensionado
    resized_clip.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    input_path = "wait_video_1.mp4"  # Ruta al video original
    output_path = "wait_video_1.mp4"  # Ruta al video redimensionado
    width = 1104  # Nuevo ancho
    height = 736  # Nuevo alto

    resize_video(input_path, output_path, width, height)
