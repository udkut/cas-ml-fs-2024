FROM python:3.12.2-slim-bookworm
LABEL maintainer "tobycheese https://github.com/tobycheese/"

# set timezone
RUN ln -sf /usr/share/zoneinfo/Europe/Zurich /etc/localtime

# install required libraries
RUN pip install --no-cache-dir jupyter jupyter_contrib_nbextensions pandas scikit-learn notebook==6.4.12 traitlets==5.9.0 ipython==8.9.0

# generate a config directory, the default config and install the exercise2 extension
RUN jupyter notebook --generate-config && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable exercise2/main

# configure jupyter for easy access (don't do this in production!!!)
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.JupyterApp.answer_yes = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_origin = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# remove unneeded stuff (for this to have any effect, the image needs to be squashed appropriately)
RUN mv /root/.local/share/jupyter/nbextensions/exercise2 . && \
    mv /root/.local/share/jupyter/nbextensions/rubberband . && \
    mv /root/.local/share/jupyter/nbextensions/collapsible_headings . && \
    rm -fr /root/.local/share/jupyter/nbextensions/* && \
    mv exercise2 rubberband collapsible_headings /root/.local/share/jupyter/nbextensions/ && \
    mv /usr/local/lib/python3.12/site-packages/jupyter_contrib_nbextensions/nbextensions/exercise2 . && \
    mv /usr/local/lib/python3.12/site-packages/jupyter_contrib_nbextensions/nbextensions/rubberband . && \
    mv /usr/local/lib/python3.12/site-packages/jupyter_contrib_nbextensions/nbextensions/collapsible_headings . && \
    rm -fr /usr/local/lib/python3.12/site-packages/jupyter_contrib_nbextensions/nbextensions/* && \
    mv exercise2 rubberband collapsible_headings /usr/local/lib/python3.12/site-packages/jupyter_contrib_nbextensions/nbextensions/

WORKDIR /notebooks
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/jupyter-notebook", "-y", "--allow-root"]
