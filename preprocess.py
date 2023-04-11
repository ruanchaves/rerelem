import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
import re
import logging 
from sklearn.model_selection import StratifiedGroupKFold
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import numpy as np 
import html

def extract_text(xml):
    soup = BeautifulSoup(xml, 'html.parser')
    return soup.get_text()

class ReRelEMReader(object):

    def __init__(self, filepath='CDSegundoHAREMReRelEM.xml'):
        self.filepath = filepath
        self.tree = ET.parse(self.filepath)
        self.root = self.tree.getroot()
    
    def extract_corel(self, text):
        if text is None:
            return None
        corel = text.split(' ')
        corel = [x for x in corel if x.strip()]
        return corel

    def extract_tiporel(self, text):
        if text is None:
            return None
        tiporel = text.split(' ')
        for idx, t in enumerate(tiporel):
            if '**' in t:
                tiporel[idx] = t.split("**")[1]
        return tiporel
    
    def find_word_indices(self, text, word_a, word_b, filter_left=False, filter_right=False, left_tag_char='<', right_tag_char='>'):
        start_tag_indices = [i for i in range(len(text)) if text.startswith(left_tag_char, i)]
        end_tag_indices = [i for i in range(len(text)) if text.startswith(right_tag_char, i)]
        a_indices = [i for i in range(len(text)) if text.startswith(word_a, i)]
        b_indices = [i for i in range(len(text)) if text.startswith(word_b, i)]

        def filter_indices(indices):
            filtered_list = []
            for index in indices:
                for list_pos, start_tag_index in enumerate(start_tag_indices):
                    if start_tag_index < index < end_tag_indices[list_pos]:
                        break
                else:
                    filtered_list.append(index)
            return filtered_list
        if filter_left:
            a_indices = filter_indices(a_indices)
        if filter_right:
            b_indices = filter_indices(b_indices)

        for a_pos, a_index in enumerate(a_indices):
            for b_pos, b_index in enumerate(b_indices):
                if b_index > a_index:
                    return b_pos
        raise ValueError(f"No occurrence of '{word_b}' after '{word_a}' in the text '{text}'")


    def get_docs(self):
        for doc in self.root.iter('DOC'):
            doc_id = doc.get('DOCID')
            for idx, p in enumerate(doc.iter('P')):

                paragraph_string = ET.tostring(p, encoding='utf8').decode('utf8')
                paragraph_text = extract_text(paragraph_string)
                
                paragraph_string = html.unescape(paragraph_string)
                
                for elem in p.iter('EM'):
                    text_index = self.find_word_indices(
                        paragraph_string, f"ID=\"{elem.get('ID')}\"", elem.text, filter_left=False, filter_right=True)
                    elem_data = {
                        'id': elem.get('ID'),
                        'text': elem.text,
                        'paragraph_text': paragraph_text,
                        'docid': doc_id,
                        'text_index': text_index
                    }
                    corel = self.extract_corel(elem.get('COREL'))
                    tiporel = self.extract_tiporel(elem.get('TIPOREL'))
                    valid_instance = isinstance(corel, list) and isinstance(tiporel, list) and len(corel) == len(tiporel)
                    if valid_instance:
                        for corel_item, tiporel_item in zip(corel, tiporel):
                            elem_data['corel'] = corel_item
                            elem_data['tiporel'] = tiporel_item
                            yield elem_data
                    else:
                        elem_data['corel'] = None
                        elem_data['tiporel'] = None
                        yield elem_data
    
    def fix_paragraph_formatting(self, text):
        text = text.strip().replace('\n', ' ')
        text = re.sub(' +', ' ', text)
        return text

    def add_tags(self, text, indexes, tags):
        indexes_list = []
        tags_list = []
        for idx, (start_span, end_span) in enumerate(indexes):
            indexes_list.append(start_span)
            tags_list.append(f"[{tags[idx]}]")
            indexes_list.append(end_span)
            tags_list.append(f"[/{tags[idx]}]")
        
        pairs = sorted(zip(indexes_list, tags_list), key=lambda x: x[0], reverse=True)
    
        for index, tag in pairs:
            text = text[:index] + tag + text[index:]

        return text

    def build_pairs(self):
        df = pd.DataFrame([row for row in self.get_docs()])
        merged_table = pd.merge(df, df, left_on='corel', right_on='id')
        records = merged_table.to_dict('records')
        for row in records:
            left_text = row['paragraph_text_x']
            right_text = row['paragraph_text_y']

            left_text = self.fix_paragraph_formatting(left_text)
            right_text = self.fix_paragraph_formatting(right_text)

            same_text = False
            if left_text == right_text:
                same_text = True
            
            left_elem_text = row['text_x']
            right_elem_text = row['text_y']

            left_docid = row['docid_x']
            right_docid = row['docid_y']

            if same_text:
                left_text_start, left_text_end = self.find_tag_indexes(left_text, left_elem_text, pos=row['text_index_x'])
                right_text_start, right_text_end = self.find_tag_indexes(left_text, right_elem_text, pos=row['text_index_y'])
                left_text = self.add_tags(
                    left_text,
                    indexes=[[left_text_start, left_text_end], [right_text_start, right_text_end]], 
                    tags=["E1", "E2"])
            else:
                left_text_start, left_text_end = self.find_tag_indexes(left_text, left_elem_text, pos=row['text_index_x'])
                right_text_start, right_text_end = self.find_tag_indexes(right_text, right_elem_text, pos=row['text_index_y'])
                left_text = self.add_tags(left_text, indexes=[[left_text_start, left_text_end]], tags=["E1"])
                right_text = self.add_tags(right_text, indexes=[[right_text_start, right_text_end]], tags=["E2"])
            
            
            assert left_docid == right_docid
            
            if same_text:
                yield {
                    'docid': left_docid,
                    'sentence1': left_text,
                    'sentence2': "",
                    'label': row['tiporel_x'],
                    'same_text': same_text,
                }
            else:
                yield {
                    'docid': left_docid,
                    'sentence1': left_text,
                    'sentence2': right_text,
                    'label': row['tiporel_x'],
                    'same_text': same_text,
                }

    def get_dataframe(self, remove_totally_nested_records=False):
        df = pd.DataFrame([row for row in self.build_pairs()])
        df.drop_duplicates(inplace=True)

        def is_totally_nested(text):
            first_case = "[E1][E2]" in text and "[/E2][/E1]" in text
            second_case = "[E2][E1]" in text and "[/E1][/E2]" in text
            third_case = "[E1][E2][/E2][/E1]" in text
            fourth_case = "[E2][E1][/E1][/E2]" in text
            fifth_case = "[E1][E2]" in text and "[/E1][/E2]" in text
            sixth_case = "[E2][E1]" in text and "[/E2][/E1]" in text
            return first_case or second_case or third_case or fourth_case or fifth_case or sixth_case

        if remove_totally_nested_records:
            df['nested'] = df['sentence1'].apply(is_totally_nested)
            df = df[df['nested'] == False]
            df.drop(columns=['nested'], inplace=True)
        return df

    def find_tag_indexes(self, text, subtext, pos=0):
        if not subtext in text:
            raise ValueError(f"Subtext '{subtext}' not found in text '{text}'")
        subtext_indices = [i for i in range(len(text)) if text.startswith(subtext, i)]
        if pos >= len(subtext_indices):
            raise ValueError(f"Subtext '{subtext}' not found in text '{text}' at position {pos}")
        start = subtext_indices[pos]
        end = start + len(subtext)
        return start, end


def stratified_k_folds(df):
    skf = StratifiedGroupKFold(n_splits=5, random_state=None, shuffle=False)
    X = np.arange(df.shape[0])
    y = df.label.values
    groups = df.docid.values
    folds = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X=X, y=y, groups=groups)):
        folds.append(test_idx)
    
    train = np.concatenate([folds[0], folds[1], folds[2]])
    test = folds[3]
    val = folds[4]

    return train, val, test


def main():
    reader = ReRelEMReader()
    
    df = reader.get_dataframe(remove_totally_nested_records=True)
    df.to_csv("rerelem.csv", index=False)
    logger.info(f"num records: {len(df)}")

    train_idx, val_idx, test_idx = stratified_k_folds(df)
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    train_df.to_csv("rerelem_train.csv", index=False)
    val_df.to_csv("rerelem_val.csv", index=False)
    test_df.to_csv("rerelem_test.csv", index=False)

    logger.info(f"train: {len(train_df)}")
    logger.info(f"val: {len(val_df)}")
    logger.info(f"test: {len(test_df)}")



if __name__ == '__main__':
    main()