# Guia para crear la libreria

instalar
~~~cmd
pip install .\whl\lib_anomaly-1.1-py2.py3-none-any.whl --force-reinstall --no-dependencies
~~~

---

## Ruta de trabajo
~~~cmd
cd "C:\Users\dherr\OneDrive - Universidad Tecnológica de Bolívar\TesisAnomDetect"
cd ".\anomaly_tesis_library"
~~~

## Identificar Python
~~~cmd
python --versión
py --version
~~~

## entorno virtual
~~~cmd
python -m venv .venv
.venv\Scripts\activate
~~~

## crear whl
~~~cmd
pip install wheel
python setup.py check sdist bdist_wheel --universal
~~~

## limpiar
~~~powershell
Move-Item -Path .\dist\* -Filter lib_anomaly-* -Destination .\whl\ -force
Remove-Item –path .\build\ -Force -Recurse
Remove-Item –path .\lib_anomaly.egg-info\ -Force -Recurse
Remove-Item –path .\dist\ -Force -Recurse
~~~

## finalizar
~~~cmd
deactivate
~~~
~~~powershell
Remove-Item –path .\.venv\ -Force -Recurse
~~~
