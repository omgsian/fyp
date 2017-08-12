# coding=utf-8

import csv

from textblob import TextBlob
from textblob.taggers import NLTKTagger
import unicodecsv
import sys

nltk_tagger = NLTKTagger()


def main(argv1, argv2):
    global data_fp, pos_file, rec, ds, msg, i, message, category, b, blob, temp_list, label, output_file, wr
    print "[log] Opening SCM file"
    data_fp = argv1
    with open('%s' % data_fp, 'rU') as pos_file:
        ds = list(tuple(rec) for rec in csv.reader(pos_file, delimiter=','))
    msg = []
    for i in range(0, len(ds)):
        message = ds[i][0]
        category = ds[i][1]
        b = unicode(message, 'utf-8')
        blob = TextBlob(b)
        temp_list = [b, category, blob.subjectivity]

        msg.append(temp_list)

    label = ["Subjective", "Objective", "Neutral"]

    for i in range(0, len(msg)):
        if msg[i][2] < 0.50:
            msg[i][2] = label[0]
        elif msg[i][2] > 0.50:
            msg[i][2] = label[1]
        elif msg[i][2] == 0.5:
            msg[i][2] = label[2]
    print '[log] Saving output to file'

    out_file = argv2
    with open(out_file, 'wb') as output_file:
        wr = unicodecsv.writer(output_file, encoding='utf-8', delimiter=',', quoting=unicodecsv.QUOTE_ALL)
        wr.writerows(msg)
    print '[log] Output file saved'


if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])