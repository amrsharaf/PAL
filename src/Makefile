unicode_to_ascii:
	iconv -c -f utf-8 -t ascii wiki.multi.en.vec > ascii.txt

main:
	python launcher_ner_bilingual.py --agent "CNNDQN" --episode 10000 --budget 1000 --train "data/conll2003/en.train;data/conll2003/en.testa;data/conll2003/en.testb;data/embeddings/muse/ascii.txt;models/en.model.saved" --test "de.train;de.testa;de.testb;de.emb;de.model.saved" --embedding_size 300
