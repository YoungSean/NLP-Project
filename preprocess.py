import pandas as pd
import tensorflow as tf

relation2label = {'Other': 0,
                  "Cause-Effect(e1,e2)": 1,
                  "Cause-Effect(e2,e1)": 2,
                  "Component-Whole(e1,e2)": 3,
                  "Component-Whole(e2,e1)": 4,
                  "Entity-Destination(e1,e2)": 5,
                  "Entity-Destination(e2,e1)": 6,
                  "Entity-Origin(e1,e2)": 7,
                  "Entity-Origin(e2,e1)": 8
                  }

label2relation = {
    0: 'Other',
    1: "Cause-Effect(e1,e2)",
    2: "Cause-Effect(e2,e1)",
    3: "Component-Whole(e1,e2)",
    4: "Component-Whole(e2,e1)",
    5: "Entity-Destination(e1,e2)",
    6: "Entity-Destination(e2,e1)",
    7: "Entity-Origin(e1,e2)",
    8: "Entity-Origin(e2,e1)"
}


class HandleDataset:
    def __init__(self, file_path, tokenizer=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.dateframe =None
        self.file2dataframe()

    def file2dataframe(self):
        with open(self.file_path, "r") as file:
            lines = [line.rstrip() for line in file]
            num_lines = len(lines)
            data = []
            for i in range(0, num_lines, 4):
                id_sentence = lines[i].split("\t")
                # sentence id
                sen_id = id_sentence[0]
                # get sentence
                sent = id_sentence[1][1:-1]
                # get the relation and replace some relation with 'other'
                relation = lines[i+1]
                if relation not in relation2label:
                    relation = 'Other'
                label = relation2label[relation]
                # remove <e1> </e1> <e2> </e2>
                sentence = sent.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                data.append([sen_id, sentence, relation, label])

            self.dateframe = pd.DataFrame(data, columns=["sentence id", "sentence", "relation", "label"])


prepocessed_data = HandleDataset("dataset/SemEval2010_task8_training/TRAIN_FILE.TXT")
print(prepocessed_data.dateframe[:5])