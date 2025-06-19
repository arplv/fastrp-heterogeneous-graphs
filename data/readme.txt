Include four areas of data derived from 20 conferences:
	Database->1
	Data Mining->2
	Machine Learning->3
	Information Retrieval->4

Dictionary files:
	author_dict.txt, conf_dict.txt, term_dict.txt; format: id+"\t"+name
Relation files:
	AA.txt, AT.txt, CA.txt, CT.txt; format: object_ID1+"\t"+object_ID2+"\t"+frequency
Label files:
	author_label.txt, conf_label.txt; format: object_id + "\t" + class_label (conf_label.txt has an additional middle column of conf name)

Example full run with the corrected code:
	python -m src.train --data data \
	                    --paths AA,A2,A3,A4,ATA,ACA \
	                    --dim 128 --epochs 50 --device cpu