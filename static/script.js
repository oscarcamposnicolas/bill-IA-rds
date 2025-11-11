// static/script.js (Versión con Modal)
document.addEventListener('DOMContentLoaded', () => {
    // Elementos de la inferencia
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const resultImage = document.getElementById('resultImage');
    const spinner = document.getElementById('spinner');

    // --- NUEVO: Elementos de la ventana modal ---
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const closeBtn = document.querySelector('.close-btn');

    // Función para abrir la modal
    function openModal(imgElement) {
        if (imgElement && imgElement.src && !imgElement.src.endsWith('/')) {
            modal.style.display = "block";
            modalImage.src = imgElement.src;
        }
    }

    // Añadir evento de clic a las imágenes "zoomables"
    imagePreview.addEventListener('click', () => openModal(imagePreview));
    resultImage.addEventListener('click', () => openModal(resultImage));

    // Añadir evento para cerrar la modal con el botón 'X'
    closeBtn.addEventListener('click', () => {
        modal.style.display = "none";
    });

    // Cerrar la modal también al hacer clic en el fondo oscuro
    modal.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });

    // Lógica para la subida y previsualización de la imagen (sin cambios)
    imageUpload.addEventListener('change', () => {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
            }
            reader.readAsDataURL(file);
            resultImage.src = "";
        }
    });

    // Lógica para la predicción (sin cambios)
    predictBtn.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            alert("Por favor, selecciona una imagen primero.");
            return;
        }

        spinner.style.display = 'block';
        resultImage.src = "";

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                resultImage.src = data.result_image + '?t=' + new Date().getTime();
            } else {
                const error = await response.json();
                alert(`Error: ${error.error}`);
            }
        } catch (error) {
            alert(`Ocurrió un error de red: ${error}`);
        } finally {
            spinner.style.display = 'none';
        }
    });
});