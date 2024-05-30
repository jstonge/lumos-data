
### Annotations

The raw data comes from the `S2ORC` collection, which is hosted on [scisciDB](https://github.com/jstonge/scisciDB). We created on our papers DB a helper collection, `papersDB.s2orc_dark_data`, containing all papers in selected venues (see `src/import/create_s2orc_dark_data.py` for details). Venues are top journals by fields identified by google scholar (top 8 categories; see [here](https://scholar.google.com/citations?view_op=top_venues&hl=en)). The coverage of the different venues vary by field, with STEM and computer science, biomed being way more represented than the social sciences and humanities. To accelerate text queries, papers were been splitted into their sections, and text was tokenized to create an index on words. Right now (2024/05/30), we have 21.4m documents/sections in the `papersDB.s2orc_dark_data`. When annotating, we take a sample from sections that contain an occurrence of `data`.

We did some annotations on excel:
 - [excel sheet](https://docs.google.com/spreadsheets/d/1PHTeYJJMRp-DpEC8oiaUH1lRjc1NTi3JDRgGj_AZojo/edit?usp=sharing)

And the related [folder](https://docs.google.com/spreadsheets/d/1PHTeYJJMRp-DpEC8oiaUH1lRjc1NTi3JDRgGj_AZojo/edit?usp=drive_link).

### Old data pipeline

 1. Pick relevant venues based on field of studies (see `journalsbyfield.csv`). 
 1. Issue 1: Metadata about venues is in openAlex, while text data is in S2ORC. We had to match venues from openAlex on venues on S2ORC, which was annoying. On 180 selected venues, we could match 170 in S2ORC. Once we know which papers we want based on venues, we created a helper collection with documents being all the sections with a mention of 'data'.

### Notes

- `2024/05/29` (JSO): I rewrote the project from scratch, using relevant scripts from `dark-dark/`. I hope the project is better organized now; we have a Makefile that keeps track of the data workflow. In the process, i realized we are missing a script that produce `clusteredDataStatements.csv`.
- `2024/05/30`(JSO) : uploaded the old annotations from our excel sheet as if they were made on label studio. That way, we can more easily see inter-annotator agreement. See `src/annotations/upload_old_annots_to_LS.py`. 