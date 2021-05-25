FROM ialcloud/compsyn-base-image:0.0.1

RUN mkdir /home/admin/comp-syn
ADD pyproject.toml poetry.lock /home/admin/comp-syn/

USER root

RUN /home/admin/.poetry/bin/poetry config virtualenvs.create false
RUN cd /home/admin/comp-syn && /home/admin/.poetry/bin/poetry install

ADD compsyn /home/admin/comp-syn/
RUN /home/admin/.poetry/bin/poetry install

WORKDIR /home/admin/comp-syn
