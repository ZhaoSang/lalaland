from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
import json
from predict import run_prediction
import fitz
from PIL import Image
import spacy
from spacy.matcher import PhraseMatcher

st.set_page_config(layout="wide")

nlp = spacy.load("en_core_web_sm")

st.cache(show_spinner=True, persist=True)

def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained("cuad-training/cuad-models/roberta-base/")
    tokenizer = AutoTokenizer.from_pretrained('cuad-training/cuad-models/roberta-base/', use_fast=False)
    return model, tokenizer


st.cache(show_spinner=True, persist=True)


def load_questions():
    with open('cuad-training/cuad-data/test.json') as json_file:
        data = json.load(json_file)

    questions = []

    for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
        question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
        questions.append(question)

    listindex = [2, 3, 5, 15]
    accountingquestions = [questions[index] for index in listindex]

    return accountingquestions


st.title("Project Rainier Demo")
basewidth = 1200
img = Image.open('somepic.jpg')
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((basewidth,hsize), Image.ANTIALIAS)
# img.save('somepic.jpg')
st.image(img, caption='Paradise at Mt Rainier National Park; photo credit: Ray Sang Jul 18, 2021')
st.subheader(
    "This demo uses CUAD AI model for contract understanding and spaCy ML model for ASC606 flagging.\nYou do not need to wait until the program to finish. You can start reviewing flagged terms almost immediately once the contract is loaded.")

add_text_sidebar = st.sidebar.title("Project Rainier")
add_text_sidebar = st.sidebar.text(
    "The Project Rainier has two functions:\n\n1. Machine learning model highlights \nterms that are relevant for \nASC 606 determination.\n\n2. AI powered general understanding \nextraction highlighting important terms \nfor pre-defined legal questions.\n \nThe current AI and machine learning \nmodels are tailored specifically \nfor Itron,Inc.")
add_text_sidebar = st.sidebar.text(
    "In honor of all the broken hearts \nresulted from reviewing lengthy \nand miserable contracts LOL")

if __name__ == '__main__':

    # Add a progress bar
    st.write('Progress Tracking')
    bar = st.progress(0)
    st.cache(show_spinner=True, persist=True)

    # add an uploader and analyze the file
    uploaded_file = st.file_uploader("Upload OCR readable pdf files only", type=['pdf'])

    if uploaded_file is not None:
        with st.spinner('Starting AI and Machine Learning computation...Be patient human...'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                contract = ""
                for page in doc:
                    contract += page.getText()
                doc.close()

            bar.progress(5)
            st.cache(show_spinner=True, persist=True)

            # show the file content uploaded
            my_expander = st.beta_expander("Contract Imported", expanded=False)
            my_expander.write(contract)

            # spacy program initiated
            # use all CPUs
            doc = nlp.pipe(contract, n_process=-1)
            doc = nlp(contract)
            bar.progress(10)
            st.cache(show_spinner=True, persist=True)

            # spacy search
            # copy below lines for each key term search
            st.subheader("Machine Learning powered ASC 606 flagging (returns BLANK if no ASC606 relevant term is found):")
            st.write("\n")
            try:
                # ML VC Search
                VariableConsideration_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                VC_phrases = ['tier price', 'price adjustment', 'liquidated damage', 'penalty', 'credit', 'discount',
                              'refund', 'bonus',
                              'not to exceed', 'price guarantee', 'price protection', 'cash', 'no cost',
                              'no additional charge']
                VC_patterns = [nlp(text) for text in VC_phrases]
                # add an ID key for the phrase_matcher
                VariableConsideration_matcher.add('VC', None, *VC_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in VariableConsideration_matcher(nlp(sent.text)):
                        st.write("VC flag:")
                        if nlp.vocab.strings[match_id] in ["VC"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(15)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Performance_Acceptance_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Performance_Acceptance_phrases = ['acceptance criteria', 'service level', 'performance guarantee',
                                                  'fan guarantee'
                                                  'coverage ratio']
                Performance_Acceptance_patterns = [nlp(text) for text in Performance_Acceptance_phrases]
                # add an ID key for the phrase_matcher
                Performance_Acceptance_matcher.add('Performance_Acceptance', None, *Performance_Acceptance_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Performance_Acceptance_matcher(nlp(sent.text)):
                        st.write("Performance or Acceptance flag:")
                        if nlp.vocab.strings[match_id] in ["Performance_Acceptance"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(20)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Enforceability_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Enforceability_phrases = ['binding forecast', 'cancel', 'early termination',
                                          'termination for convenience',
                                          'nonrefundable', 'non-refundable']
                Enforceability_patterns = [nlp(text) for text in Enforceability_phrases]
                # add an ID key for the phrase_matcher
                Enforceability_matcher.add('Enforceability', None, *Enforceability_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Enforceability_matcher(nlp(sent.text)):
                        st.write("Enforceability flag:")
                        if nlp.vocab.strings[match_id] in ["Enforceability"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(25)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Transfer_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Transfer_phrases = ['FOB destination', 'FOB shipping point', 'title', 'ownership', 'risk of loss'
                                                                                                   'FOB origin',
                                    'shipment term',
                                    'shipping term', 'bill and hold']
                Transfer_patterns = [nlp(text) for text in Transfer_phrases]
                # add an ID key for the phrase_matcher
                Transfer_matcher.add('Transfer', None, *Transfer_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Transfer_matcher(nlp(sent.text)):
                        st.write("Transfer flag:")
                        if nlp.vocab.strings[match_id] in ["Transfer"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(30)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                MR_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                MR_phrases = ['option', 'right to purchase', 'to be determined', 'TBD', 'upgrade', 'firmware update',
                              '5G update',
                              'regulatory approval', 'state approval', 'government approval', 'agency approval',
                              'council approval']
                MR_patterns = [nlp(text) for text in MR_phrases]
                # add an ID key for the phrase_matcher
                MR_matcher.add('MR', None, *MR_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in MR_matcher(nlp(sent.text)):
                        st.write("Option, material right and other rights flag:")
                        if nlp.vocab.strings[match_id] in ["MR"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(35)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Warranty_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Warranty_phrases = ['warranty period', 'extended warranty', 'standard warranty', 'third party warranty']
                Warranty_patterns = [nlp(text) for text in Warranty_phrases]
                # add an ID key for the phrase_matcher
                Warranty_matcher.add('Warranty', None, *Warranty_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Warranty_matcher(nlp(sent.text)):
                        st.write("Warranty flag:")
                        if nlp.vocab.strings[match_id] in ["Warranty"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(40)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Pmt_Term_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Pmt_Term_phrases = ['payment', 'invoice']
                Pmt_Term_patterns = [nlp(text) for text in Pmt_Term_phrases]
                # add an ID key for the phrase_matcher
                Pmt_Term_matcher.add('Payment', None, *Pmt_Term_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Pmt_Term_matcher(nlp(sent.text)):
                        st.write("Payment Term flag:")
                        if nlp.vocab.strings[match_id] in ["Payment"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(45)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Right_of_Return_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Right_of_Return_phrases = ['refund', 'return', 'rework', 're-work', 'exchange', 'Repurchase']
                Right_of_Return_patterns = [nlp(text) for text in Right_of_Return_phrases]
                # add an ID key for the phrase_matcher
                Right_of_Return_matcher.add('Right_of_Return', None, *Right_of_Return_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Right_of_Return_matcher(nlp(sent.text)):
                        st.write("Right of Return, Rework or Repurchase flag:")
                        if nlp.vocab.strings[match_id] in ["Right_of_Return"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(50)
                st.cache(show_spinner=True, persist=True)
                License_Patent_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                License_Patent_phrases = ['right to use', 'license', 'patent', 'right to access']
                License_Patent_patterns = [nlp(text) for text in License_Patent_phrases]
                # add an ID key for the phrase_matcher
                License_Patent_matcher.add('License_Patent', None, *License_Patent_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in License_Patent_matcher(nlp(sent.text)):
                        st.write("License, patent or access flag:")
                        if nlp.vocab.strings[match_id] in ["License_Patent"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(55)
                st.cache(show_spinner=True, persist=True)
                Principal_Agent_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Principal_Agent_phrases = ['agent', 'principal', 'third party', 'subcontractor', 'supplier', 'vendor',
                                           'RACI']
                Principal_Agent_patterns = [nlp(text) for text in Principal_Agent_phrases]
                # add an ID key for the phrase_matcher
                Principal_Agent_matcher.add('Principal_Agent', None, *Principal_Agent_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Principal_Agent_matcher(nlp(sent.text)):
                        st.write("Principal vs Agent flag:")
                        if nlp.vocab.strings[match_id] in ["Principal_Agent"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(60)
                st.cache(show_spinner=True, persist=True)
                Related_Agreement_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Related_Agreement_phrases = ['side agreement', 'vendor agreement', 'vendor SOW',
                                             'installation agreement',
                                             'installation SOW', 'subcontractor agreement', 'subcontractor SOW',
                                             'loan document',
                                             'lease document', 'financing document', 'loan agreement',
                                             'lease agreement',
                                             'financing agreement', 'csa', 'customer specific addendum', 'addendum']
                Related_Agreement_patterns = [nlp(text) for text in Related_Agreement_phrases]
                # add an ID key for the phrase_matcher
                Related_Agreement_matcher.add('Related_Agreement', None, *Related_Agreement_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Related_Agreement_matcher(nlp(sent.text)):
                        st.write("Related Agreements flag:")
                        if nlp.vocab.strings[match_id] in ["Related_Agreement"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(65)
                st.cache(show_spinner=True, persist=True)
                # copy below lines for each key term search
                Retention_Bond_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Retention_Bond_phrases = ['retention amount', 'bond', 'withhold']
                Retention_Bond_patterns = [nlp(text) for text in Retention_Bond_phrases]
                # add an ID key for the phrase_matcher
                Retention_Bond_matcher.add('Retention_Bond', None, *Retention_Bond_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Retention_Bond_matcher(nlp(sent.text)):
                        st.write("Retention or Bond flag:")
                        if nlp.vocab.strings[match_id] in ["Retention_Bond"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(70)
                st.cache(show_spinner=True, persist=True)
                # ML search other matters
                Other_Matters_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
                Other_Matters_phrases = ['tax', 'reimburse', 'indemnify', 'term of']
                Other_Matters_patterns = [nlp(text) for text in Other_Matters_phrases]
                # add an ID key for the phrase_matcher
                Other_Matters_matcher.add('Other_Matters', None, *Other_Matters_patterns)
                # replace the following for new key term search
                for sent in doc.sents:
                    for match_id, start, end in Other_Matters_matcher(nlp(sent.text)):
                        st.write("Other_Matters (e.g., expense, taxes, indemnification,contract terms, etc.) flag:")
                        if nlp.vocab.strings[match_id] in ["Other_Matters"]:
                            st.write(sent.text)
                            st.write(
                                "------------------------------------------------------------------------------------")
                bar.progress(75)
                st.cache(show_spinner=True, persist=True)
            except:
                st.write("Machine Learning model has not flagged any ASC606 relevant terms. Check contract imported.")

            # AI STARTS HERE!!!!!!!!!!!!!!!!!!!upload models and ASC 606 questions


            st.subheader("AI powered overall contract review section (returns BLANK if no relevant question is found):")
            st.write("Warning: AI may not be accurate so please exercise your due diligence and care.")
            st.write("\n")
            model, tokenizer = load_model()
            bar.progress(80)
            st.cache(show_spinner=True, persist=True)

            questions = load_questions()
            bar.progress(85)
            st.cache(show_spinner=True, persist=True)

            # run predictions
            try:
                index = 1
                prediction = run_prediction(questions[:], contract, 'cuad-training/cuad-models/roberta-base/')
                bar.progress(95)
                st.cache(show_spinner=True, persist=True)

                answers = list(prediction.values())
                bar.progress(99)
                st.cache(show_spinner=True, persist=True)

                # only write out questions and answers if an answer is found

                for question in questions:
                    if len(answers[index - 1]) != 0:
                        st.write(question)
                        st.write("AI found: " + answers[index - 1])
                        index += 1
                    else:
                        index += 1
            except:
                st.write(
                    "Nothing is found by the AI. Check 'Contract Imported' to see whether contract has been correctly imported or not.")
            bar.progress(100)
            st.subheader("Congrats we finished the analysis together!")
            st.cache(show_spinner=True, persist=True)
            st.balloons()
            contract = ""
    else:
        with st.spinner('You can feed me some readable pdf contracts...'):
            pass
