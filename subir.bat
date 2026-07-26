@echo off
echo Agregando cambios...
git add .
echo Guardando cambios...
set /p mensaje="Escribe el mensaje de tu cambio: "
git commit -m "%mensaje%"
echo Subiendo a GitHub y Render...
git push origin main
echo ¡Listo! Todo actualizado con exito.
pause