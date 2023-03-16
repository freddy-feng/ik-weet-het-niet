import pandas as pd
import numpy as np
from datetime import datetime
import random

class Deck:
    def __init__(self, word_list=None):
        # Use the word as unique keys
        if word_list is None:
            self.word_list = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        else:
            self.word_list = word_list
        self.df = pd.DataFrame({'key': self.word_list})
        self.df.set_index('key')
        
        self.df['link'] = ['https://www.woorden.org/woord/' + w for w in self.word_list]
        
        # Visit stat
        self.df['n_visit'] = 0
        self.df['last_visit'] = datetime.now()
        self.df['study_interval'] = 0

        # For Leitner 10+2 box system
        ## Session 0, 1, ..., 9 to match with 10 boxes
        ## Max box number is limited by digits
        ## Can expand if using alphabet naming
        self.df['is_current'] = True
        self.df['is_retired'] = False
        self.previous_session = 9

        # 10 boxes: 0259, 1360, 2471, ... 
        # 4 tiers, current -> tier1...4 -> retire
        n1 = np.arange(0, 10)
        n2 = (n1 + 2) % 10
        n3 = (n1 + 5) % 10
        n4 = (n1 + 9) % 10

        x = pd.DataFrame([n1, n2, n3, n4]).transpose()
        self.box_ids = x[0].astype("string") + x[1].astype("string") + x[2].astype("string") + x[3].astype("string")
        self.df['box_id'] = [random.choice(self.box_ids) for _ in range(len(self.word_list))]

        # For SM-2 alg
        self.score_dict = {
            0: 'Total blackout',
            1: 'Incorrect response, but upon seeing the correct answer it felt familiar.',
            2: 'Incorrect response, but upon seeing the correct answer it seemed easy to remember.',
            3: 'Correct response, but required significant effort to recall.',
            4: 'Correct response, after some hesitation.',
            5: 'Correct response with perfect recall.'
        }
        self.df['easiness'] = 2.5 # Default acc to SM-2 wiki
        self.df['n_scores'] = [{g: 0 for g in range(6)} for _ in range(len(self.word_list))]


    def update_df(self, session, key, score):
    # This function is used to update the self.df instance when
    # the user has answered a question/key (played a card)
        session = str(session)
        is_key = self.df['key'] == key
    
        self.df.loc[is_key, 'n_visit'] += 1
        self.df.loc[is_key, 'last_visit'] = datetime.now()

        if (score >= 0) & (score < 3):
            self.df[is_key]['n_scores'].iloc[0][score] += 1
            self.df.loc[is_key, 'is_current'] = True # Incorrect. Stay in current box.
            self.df.loc[is_key, 'is_retired'] = False
        elif (score >= 3) & (score <= 5):
            # Correct
            self.df[is_key]['n_scores'].iloc[0][score] += 1
            self.df.loc[is_key, 'is_current'] = False # Review later
            self.change_box(session, key, score)
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
        b1 = self.df.loc[is_key, 'box_id'].iloc[0]
        
        
        if session == list(b1)[0]: # Check first digit
            # Move to next tier
            b2 = self.get_box_id_equal_session(session, b1, self.box_ids)
            self.df.loc[is_key, 'box_id'] = b2
            return None
        if (session == list(b1)[-1]) & (score == 5):
            # Move to retirement
            self.df.loc[is_key, 'is_retired'] = True # Correct. Revisit later.
            return None
        else:
            self.df.loc[is_key, 'box_id'] = b1 # Stay
            return None
        
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
        t = b.index(s)
        t2 = t + 1
        if t2 > 3: # max tier is 3 (since 0,1,2,3)
            return np.NaN # Retire
        else:
            items2 = self.get_box_ids_contain_s(s, items) # Shortlist box ids matching current session
            return [item for item in items2 if t2 == item.index(s)][0]

            
    # Update interval for sm2
    # Update easiness for sm2
        