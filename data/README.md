# Data access

The full SomBench NAC image dataset is not included in this repository. Retrieve it from the NASA–IBM AI4Science Hugging Face dataset linked in the main README. The preparation script expects the downloaded files at `work/data/Sombench-NAC-Crater-Detection/`.

Apollo PSE records are NASA PDS data (bundle DOI 10.17189/9ykc-er91; network DOI 10.7914/SN/XA_1969). The impact downloader script retrieves the four S14 event records. The KO-26/KO-40 comparison requires the S15 MiniSEED and StationXML files; refer to the public archive and the processing summary before running. Raw data are not committed here.
