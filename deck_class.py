import pandas as pd
import numpy as np
from datetime import datetime
import random

class Deck:
    def __init__(
        self, 
        word_list=['Hearts', 'Diamonds', 'Clubs', 'Spades', 'Joker']
    ):
        
        # Use the word as unique keys
#        if word_list is None:
#            self.word_list = ['Hearts', 'Diamonds', 'Clubs', 'Spades', 'Joker']
#        else:
#            self.word_list = word_list
    
    
        # Create dataframe from the word list
#        self.df = pd.DataFrame({'key': self.word_list})
#        self.df['link'] = ['https://www.woorden.org/woord/' + w for w in self.word_list]
    
        self.df = pd.DataFrame({'key': word_list})
        self.df['link'] = ['https://www.woorden.org/woord/' + w for w in word_list]

        # Visit stat
        self.df['n_visit'] = 0
        self.df['study_interval'] = 0

        # For Leitner 10+2 box system
        ## Session 0, 1, ..., 9 to match with 10 boxes
        ## Max box number is limited by digits
        ## Can expand if using alphabet naming
        self.df['is_current'] = True
        self.df['is_retired'] = False
        
        self.df['last_score'] = 0
        self.df['last_visit'] = datetime.now()
        self.df['last_session'] = 'J'

        # 10 boxes: 0259, 1360, 2471, ... 
        # 4 tiers, current -> tier1...4 -> retire
        n1 = np.arange(0, 10)
        n2 = (n1 + 2) % 10
        n3 = (n1 + 5) % 10
        n4 = (n1 + 9) % 10
        
        # Alphabet verion
        n1 = [chr(x+97).upper() for x in n1]
        n2 = [n1[x] for x in n2]
        n3 = [n1[x] for x in n3]
        n4 = [n1[x] for x in n4]

        x = pd.DataFrame([n1, n2, n3, n4]).transpose()
        self.box_ids = x[0].astype('string') + x[1].astype('string') + x[2].astype('string') + x[3].astype('string')
        self.df['box_id'] = [random.choice(self.box_ids) for _ in range(len(self.df['key']))]

        # For SM-2 alg
        score_dict = {
            's0': 'Total blackout',
            's1': 'Incorrect response, but upon seeing the correct answer it felt familiar.',
            's2': 'Incorrect response, but upon seeing the correct answer it seemed easy to remember.',
            's3': 'Correct response, but required significant effort to recall.',
            's4': 'Correct response, after some hesitation.',
            's5': 'Correct response with perfect recall.'
        }
        
        self.df['easiness'] = 2.5 # Default acc to SM-2 wiki
        self.df['n_scores'] = [{'s'+str(g): 0 for g in range(6)} for _ in range(len(self.df['key']))]


        
        
    def copy_df(self, df):
    # Load old deck csv to current deck
        self.df = df
#        self.word_list = self.df['key'].tolist()
#        self.df = self.df.set_index('key')
        # Set data to correct format
        self.df['last_visit'] = pd.to_datetime(self.df['last_visit']) # string to datetime for visit time
        self.df['box_id'] = self.df['box_id'].astype(str) # integer to string for box id
        self.df['n_scores'] = [eval(x) for x in self.df['n_scores']] # string to tuple for n_scores
        self.df['n_scores'] = [{str(k): v for (k,v) in item.items()} for item in self.df['n_scores']] # Change key to str


    def update_df(self, session, key, score):
    # This function is used to update the self.df instance when
    # the user has answered a question/key (played a card)
        session = str(session)
        is_key = self.df['key'] == key
    
        self.df.loc[is_key, 'n_visit'] += 1
        self.df.loc[is_key, 'last_visit'] = datetime.now()
        self.df.loc[is_key, 'last_score'] = score
        self.df.loc[is_key, 'last_session'] = session

        if (score >= 0) & (score < 3):
            self.df.loc[is_key, 'n_scores'].iat[0]['s'+str(score)] += 1 # score is the key in the n_score dict
            self.df.loc[is_key, 'is_current'] = True # Incorrect. Stay in current box.
            self.df.loc[is_key, 'is_retired'] = False
        elif (score >= 3) & (score <= 5):
            # Correct
            self.df.loc[is_key, 'n_scores'].iat[0]['s'+str(score)] += 1
            self.df.loc[is_key, 'is_current'] = False # Review later
            self.change_box(session, key, score) # Check if promote to next tier
        else:
            raise ValueError('Score must be [0, 1, ..., 5]')

            
    def change_box(self, session, key, score):
    # Function to decide the next box placement
    # In the Leitner system, correctly answered cards are advanced to the next, less frequent box
    # incorrectly answered cards return to the current box.
    # Either move to next tier if session matches FIRST digit of box id
    # or move to retirement if session matches LAST digit of box id and get full score 5
        session = str(session)
        is_key = self.df['key'] == key
        b1 = self.df.loc[is_key, 'box_id'].iat[0] # Box id for the unqiue key, string
        
        if session == list(b1)[0]: # Check first digit
            # Move to next tier
            b2 = self.get_box_id_equal_session(session, b1, self.box_ids) 
            self.df.loc[is_key, 'box_id'] = b2

        if (session == list(b1)[-1]) & (score == 5):
            # Move to retirement
            self.df.loc[is_key, 'is_retired'] = True # Correct. Revisit later.

        else:
            self.df.loc[is_key, 'box_id'] = b1 # Stay

        
    def get_box_ids_contain_s(self, s, items):
    # Return list of box ids that contains session number
    # s: current session
    # items: complete list of box ids
        s = str(s)
        return [item for item in items if s in item]

    def get_box_id_equal_session(self, s, b, items):
    # Return the box id for next tier 
    # s: current session
    # b: current box id
    # items: complete list of box ids
        s = str(s)
        t = b.index(s) # Get the index of s in string b
        t2 = t + 1 # Index + 1 -> next tier
        if t2 > 3: # max tier is 3 (0,1,2,3 for 4-digit box id)
            return np.NaN # The key can retire
        else:
            items2 = self.get_box_ids_contain_s(s, items) # Shortlist box ids matching current session
            return [item for item in items2 if t2 == item.index(s)][0]


    def scores_dict_to_df(self, drop_scores_dict=False):
        # Convert the n_scores in dict format five columns and right join to deck df
        # Return a new df 
        
        # A scores dict per key changed to dataframe
        scores = pd.concat([pd.DataFrame(x, index=[0]) for x in self.df['n_scores']]) 
        # Rename columns
        scores = scores.rename(columns={i: 'score'+str(i) for i in range(6)}).reset_index(drop=True)
        # Join new df with original df
        df = self.df.join(scores, how='left')
        
        if drop_scores_dict: # Drop n_scores column
            df.drop(['n_scores'], axis=1, inplace=True)
        return df
        
    # Update interval for sm2
"""
# NOT USED 
    def update_easiness(self, easiness, n):
    # SM-2 algorithm to update easiness factor
    # Algorithm is adapted from
    # https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
        q = 0
        if n in [0, 1]:
            q = 0
        elif n == 2:
            q = 1
        elif n in [3, 4]:
            q = 2
        elif n == 5:
            q = 3

        ef = easiness + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        if ef < 1.3:
            ef = 1.3
        if ef > 2.5:
            ef = 2.5
        return ef
"""