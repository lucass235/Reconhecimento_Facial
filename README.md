# Reconhecimento_Facial

## Descrição

- Esse projeto consiste em um sistema de reconhecimento facial, onde o usuário pode cadastrar uma nova pessoa, treinar o modelo e reconhecer a pessoa cadastrada. A ideia é que esse sistema seja utilizado para controlar o acesso de pessoas em um determinado local, como por exemplo, uma empresa ou um condomínio. O sistema foi desenvolvido utilizando a linguagem Python e a biblioteca OpenCV e face_recognition.

## Como Funciona

- O sistema funciona da seguinte forma: Primeiro o usuário deve cadastrar uma nova pessoa, para isso ele deve informar o nome da pessoa e tirar uma foto dela. Após isso, o usuário deve treinar o modelo, para que o modelo aprenda a reconhecer a pessoa cadastrada. Por fim, o usuário pode reconhecer a pessoa cadastrada, para isso ele deve tirar uma foto da pessoa e o sistema irá informar se a pessoa é ou não a pessoa cadastrada. Depois é só repetir o processo para cadastrar novas pessoas.

## Interface Gráfica

- O sistema por enquanto possui apenas a interface da webCam, onde o usuário pode ver seu rosto e é fornecido um feedback para o usuário, informando se o rosto está sendo reconhecido ou não, neste caso, se a pessoa esta com acesso liberado ou não.

## Como Executar

- Para executar o sistema, basta executar o arquivo webcan_main.py, que está dentro da pasta ./face_recognition/ para isso, basta executar o seguinte comando no terminal dentro da pasta:

```bash
python webcan_main.py
```

- Após isso, o sistema irá abrir uma janela com a interface da webCam, onde o usuário pode ver seu rosto.

## Bibliotecas Utilizadas

- OpenCV
- face_recognition

    - Para instalar as bibliotecas, basta executar o seguinte comando no terminal:

```bash
pip install -r requirements.txt
```

OBS: Para rodar o sistema utilizando uma GPU da NVIDIA, é necessário instalar o CUDA Toolkit e a biblioteca especifica do opencv para GPU que esta dentro do arquivo requirements.txt, (opencv-python-headless), para isso é necessário descometa a linha referente a essa biblioteca dentro do arquivo requirements.txt e executar o comando acima.

No windows voce pode ter problemas para baixar a biblioteca face_recognition, para isso, basta seguir o tutorial abaixo:

- Voce deve instalar a biblioteca dlib manualmente no terminal 

- O arquivo da biblioteca esta dentro da pasta ./instalaçao_dlib/

- Verifique a sua versão do pyhton e baixe a biblioteca dlib correspondente

- Para instalar a biblioteca dlib, basta executar o comando `pip install <Arquivo dlib da sua versao do python correspondente>`. Exemplo: `pip install dlib-19.21.1-cp37-cp37m-win_amd64.whl`.