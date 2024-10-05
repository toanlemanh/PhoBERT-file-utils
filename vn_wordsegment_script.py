import py_vncorenlp
import os
import re
from helpers.check_models import check_models
from helpers.reformat_wordsegment_output import reformat_wordsegment_output as process_wordsegment

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
def vn_nlp():
    abspath = os.path.abspath(".")
    print(abspath)
    input_file = abspath + '/output/output_corpus.txt'
    output_file = abspath +'/output/pre_output.txt'
    wordsegment_file = abspath + "/output/final_output.txt"
    file_path = abspath +"/VNCoreNLP-1.2.jar"
    dir_path = abspath +"/models"
    if ( not check_models(file_path, dir_path) ) :
        py_vncorenlp.download_model(save_dir= abspath + '/vncorenlp')

    # Load the word and sentence segmentation component
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir = abspath + '/vncorenlp')

    # Receive the corpus output file (after corpus reader) and generate a word_segment file
    rdrsegmenter.annotate_file(input_file= input_file, output_file= output_file)

    # Reformat the content in file and generate a final file
    return process_wordsegment(input_file_path=output_file, output_file_path= wordsegment_file)
vn_nlp()