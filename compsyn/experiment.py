# consume list of dictionaries, create a WordToColorVector for each one

header = "search_term	Bigram	Conc.M	Conc.SD	Unknown	Total	Percent_known	SUBTLEX	Dom_Pos"
rows = """
solid	0	4.42	0.81	0	26	1	998	Adjective
woolly	0	3.96	1.14	1	26	0.96	24	Adjective
brevity	0	2.23	1.31	3	29	0.9	11	Noun
"""

experiment = list()
for row in rows.split("\n"):
    values = row.split()
    experiment.append({key: val for key, val in zip(keys, values)})

print(experiment)
