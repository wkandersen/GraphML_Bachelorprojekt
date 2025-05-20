Alting skal køres fra root mappen (cd GraphML_Bachelorprojekt)

1. Download datasættet ved hjælp af data.py
2. Kør script1_dataset.sh, der laver en dictionary med random embeddings for træningsdata
3. Kør derefter script2_batches.sh for at se hvordan vi træner embeddings 
   (sørg altid for at embedding_dim er det samme for de to scripts)

Hvis man kun vil køre for et par batches skal num_iterations linje 69 i embed_batches_2.py ændres til det ønskede antal batches.
Epochs kan ændres i script2_batches.sh. 