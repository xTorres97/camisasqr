AR Camiseta - Paquete mínimo
============================

Contenido:
- target.png       -> Imagen optimizada para generar el archivo targets.mind (usa esta imagen en MindAR Studio).
- index.html       -> Página WebAR lista para usar. Coloca aqui el archivo 'targets.mind' y opcionalmente 'model.glb'.
- (Opcional) model.glb -> tu modelo 3D si lo añades; el archivo por defecto espera ./model.glb

Pasos para finalizar (generar targets.mind y publicar):
1) Generar 'targets.mind' usando MindAR Studio (online):
   - Abre MindAR Studio (busca "MindAR Studio" o revisa la documentación de MindAR).
   - Sube 'target.png' (este archivo) y genera el paquete. Descarga el archivo 'targets.mind' que produzca.
   - Renombra y coloca 'targets.mind' en la misma carpeta que index.html.

2) Añadir tu modelo (opcional):
   - Coloca tu modelo glTF/GLB en la misma carpeta y nómbralo 'model.glb', o modifica index.html para apuntar al nombre correcto.

3) Probar localmente (recomendado usar un servidor local con HTTPS):
   - GitHub Pages o Netlify son opciones gratuitas y ya tienen HTTPS.
   - Alternativa local: usa 'live-server' y confía en certificado (más complejo en móvil).

4) Publicar en GitHub Pages:
   - Crea un repositorio, sube todo el contenido de esta carpeta y habilita GitHub Pages en Settings -> Pages -> Branch main / root.
   - Tras publicar, copia la URL (https://<tu-usuario>.github.io/<repo>/).
   - Genera un QR que apunte a esa URL usando cualquier generador de QR gratuito.

5) Probar en móvil:
   - Abre el QR o la URL en Chrome (Android) o Safari (iOS). Dale permiso a la cámara.
   - Apunta a la camiseta con la imagen completa. Si la detección falla, ajusta iluminación/distancia.

Notas:
- Si quieres, puedo guiarte paso a paso cómo usar MindAR Studio para generar 'targets.mind' y ayudarte a subir a GitHub Pages.
- Si prefieres, también puedo generar una versión ZIP de este paquete para que la descargues directamente.
