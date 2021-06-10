FROM ialcloud/comp-syn-base:1.0.0

RUN mkdir /home/admin/comp-syn
ADD requirements.txt /home/admin/comp-syn/

USER root

RUN python3.8 -m pip install -r /home/admin/comp-syn/requirements.txt

ADD compsyn /home/admin/comp-syn/
RUN cd /home/admin/comp-syn && python3.8 -m pip install .

WORKDIR /home/admin/comp-syn
