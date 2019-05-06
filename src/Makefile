unicode_to_ascii:
	iconv -c -f utf-8 -t ascii wiki.multi.en.vec > ascii.txt

main:
	python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10000 --budget 1000 --train "data/conll2003/en.train;data/conll2003/en.testa;data/conll2003/en.testb;data/embeddings/muse/ascii.txt;models/en.model.saved" --test "de.train;de.testa;de.testb;de.emb;de.model.saved" --embedding_size 300

main_es:
	python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10 --budget 10 --train "data/conll2002/es.train;data/conll2002/es.testa;data/conll2002/es.testb;data/embeddings/cca/es.emb;models/es.model.saved" --test "de.train;de.testa;de.testb;de.emb;de.model.saved" --embedding_size 40

main_nl:
	python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10 --budget 10 --train "data/conll2002/nl.train;data/conll2002/nl.testa;data/conll2002/nl.testb;data/embeddings/cca/nl.emb;models/nl.model.saved" --test "data/conll2003/de.train;data/conll2003/de.testa;data/conll2003/de.testb;data/embeddings/cca/de.emb;models/de.model.saved" --embedding_size 40

main_en:
	python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10000 --budget 200 --train "data/conll2003/en.train;data/conll2003/en.testa;data/conll2003/en.testb;data/embeddings/cca/en.emb;models/en.model.saved" --test "data/conll2003/de.train;data/conll2003/de.testa;data/conll2003/de.testb;data/embeddings/cca/de.emb;models/de.model.saved" --embedding_size 40

main_en_gpu:
	python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10000 --budget 200 --train "/scratch0/PAL/conll2003/en.train;/scratch0/PAL/conll2003/en.testa;/scratch0/PAL/conll2003/en.testb;/scratch0/PAL/embeddings/cca/en.emb;/scratch0/PAL/models/en.model.saved" --test "/scratch0/PAL/conll2003/de.train;/scratch0/PAL/conll2003/de.testa;/scratch0/PAL/conll2003/de.testb;/scratch0/PAL/embeddings/cca/de.emb;/scratch0/PAL/models/de.model.saved" --embedding_size 40

split:
	python split_embedddings.py --language nl --input_path data/embeddings/CCA/fifty_nine.table5.multiCCA.size_40.normalized --output_path data/embeddings/CCA/nl.emb

preprocess_en:
	python preprocess.py -i data/conll2003/eng.train -o data/conll2003/en.train;
	python preprocess.py -i data/conll2003/eng.testa -o data/conll2003/en.testa;
	python preprocess.py -i data/conll2003/eng.testb -o data/conll2003/en.testb;

preprocess_es:
	python preprocess.py -i data/conll2002/esp.train -o data/conll2002/es.train;
	python preprocess.py -i data/conll2002/esp.testa -o data/conll2002/es.testa;
	python preprocess.py -i data/conll2002/esp.testb -o data/conll2002/es.testb;

preprocess_nl:
	python preprocess.py -i data/conll2002/ned.train -o data/conll2002/nl.train;
	python preprocess.py -i data/conll2002/ned.testa -o data/conll2002/nl.testa;
	python preprocess.py -i data/conll2002/ned.testb -o data/conll2002/nl.testb;

preprocess_de:
	python preprocess.py -i data/conll2003/deu.train -o data/conll2003/de.train;
	python preprocess.py -i data/conll2003/deu.testa -o data/conll2003/de.testa;
	python preprocess.py -i data/conll2003/deu.testb -o data/conll2003/de.testb;

