# Make .whl

## 1. Preparar un entorno virtual
~~~ powershell
python --version
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.venv\Scripts\activate
~~~

## 2. instalar librerias
~~~ powershell
pip install wheel
~~~

## 3. Crear libreria de tesis
~~~ powershell
cd anomaly_tesis_library
pip uninstall -y lib_anomaly
python setup.py check sdist bdist_wheel --universal
Move-Item -Path .\dist\* -Filter lib_anomaly-* -Destination .\whl\ -force
Remove-Item –path .\build\ -Force -Recurse
Remove-Item –path .\lib_anomaly.egg-info\ -Force -Recurse
Remove-Item –path .\dist\ -Force -Recurse
cd..
pip install .\anomaly_tesis_library\whl\lib_anomaly-0.1-py2.py3-none-any.whl --force-reinstall --no-dependencies
~~~
