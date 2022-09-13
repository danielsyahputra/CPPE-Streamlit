# Medical Personal Protective Equipment Detection

[![Source Code](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1DlRUOuUZrDrEhnkIJFo-ilAfVrKsz5k9?usp=sharing)
[![Docker Image](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/danielsyahputra13/cppe_streamlit)

This project is implemented using the CPPE-5 Dataset and modified based on research done by Rishit Dagli and Ali Mustufa Shaikh in their famous paper _**CPPE - 5: Medical Personal Protective Equipment Dataset**_.

_**Accompanying paper: [CPPE - 5: Medical Personal Protective Equipment Dataset](https://arxiv.org/abs/2112.09569)**_

## About This App

### Labels
| Label | Description |
|:----:|:-------------|
| 0 | Coverall |
| 1 | Face_Shield |
| 2 | Gloves |
| 3 | Goggles |
| 4 | Mask |

## Run This App

### Repository Clone
```
git clone https://github.com/danielsyahputra13/CPPE-Streamlit.git
cd CPPE-Streamlit
python3 download.py
pip3 install -r requirements.txt
streamlit run interface/App.py
```

### Docker
The image is avaliable at [Docker hub](https://hub.docker.com/repository/docker/danielsyahputra13/cppe_streamlit). To run this app, you can do the following commands.

- Pull image: `docker pull danielsyahputra13/cppe_streamlit:stable`
- Run docker container locally: `docker run -p {LOCAL_PORT}:8501 danielsyahputra13/cppe_streamlit:stable`
- Open the browser and go to `localhost:{LOCAL_PORT}` to see the application.

## Acknoweldgements

```
- Dagli, Rishit, and Ali Mustufa Shaikh. “CPPE-5: Medical Personal Protective Equipment Dataset.” ArXiv.org, 15 Dec. 2021, https://arxiv.org/abs/2112.09569. 
- Dagli, R., & Shaikh, A. M. (2021). arXiv:2112.09569. arXiv:2112.09569
```
