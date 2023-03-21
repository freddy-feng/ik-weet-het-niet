###############################################################################
# Dependencies
import os
import random
from pathlib import Path
import copy
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64

import matplotlib.pyplot as plt
import seaborn as sns

import deck_class
###############################################################################
debug = False
#st.set_page_config()

###############################################################################
# Functions
@st.cache_data
def fetch_sample_data():
    csv_file_path = Path('data/word_freq_dutch-news-articles-1000.csv')
    data = pd.read_csv(csv_file_path)
    return data

def create_new_deck(mode='simple_new', file_path='data/card_deck.csv'):   
    # return a new deck class object
    # mode: create 'new' deck or 'load' previous df fromd csv
    # 'simple_new' uses the default cards five suits for demonstration

    if mode == 'new':
        # Create data for new user
        # Create a sample deck using 1000 dutch words
        data = fetch_sample_data()
        word_list = data['string'].tolist()
        deck2 = deck_class.Deck(word_list) # Create deck object

    elif mode == 'load':
        # Load deck csv from path
        file_path = Path(file_path)
        if file_path.is_file():
            # file exists
            data = pd.read_csv(file_path) # Previous df in csv
            deck2 = deck_class.Deck() # Create deck object
            deck2.copy_df(data) # Load df and formatting
        else:
            raise ValueError('File does not exist!')
        
    elif mode == 'simple_new':
        deck2 = deck_class.Deck()
    
    return deck2

def set_up_practice(deck, q_count, session):
    # Read user requirement to randomly pick the cards for practice
    
    # Get box ids containing the session number
    review_boxes = [b for b in deck.df['box_id'] if str(session) in list(b)]
    is_review = [b1 in review_boxes for b1 in deck.df['box_id']]

    # Cards from review boxes set to current deck
    deck.df.loc[is_review, 'is_current'] = True # Update deck
    
    current_cards = [x for x in deck.df.loc[deck.df['is_current'], 'key']]

    q_max = len(current_cards)
    if q_count > q_max:
        # Not enough current cards for practice
        st.warning(f'Reduce number of questions to {q_max} because of too few current cards!')
        q_count = q_max

    practice_cards = random.sample(current_cards, q_count)

    return deck, practice_cards, session, q_count


def get_slider_score(deck, key, session, i):
    help_score = '5 (breeze), 4 (ok + hesitate), 3 (ok + sweat), 2 (no but quite familiar), 1 (no but still familiar), 0 (blackout)'
    # The key of the slider, get the session state of slider is the score
    st.slider(
        'slider_score', label_visibility='hidden',
        min_value=0, max_value=5, step=1,            
        key='q'+str(i), help=help_score
    )
#    return st.session_state['q'+str(i)] 


def show_one_question(deck, practice_cards, session, q_count, i):
    # For each question, display question number, question, slider to get user score
    key = practice_cards[i] # Question word
    col1, col2, col3, col4 = st.columns([2, 4, 3, 0.5])
    with col1:
    # Display question number
        str_q = f"""<div style="text-align: center;"><br>{i+1} of {q_count}</div>"""
        st.markdown(str_q, unsafe_allow_html=True)
    with col2:
    # Display question
        st.header(f'__{key}__')
        is_key = deck.df['key'] == key
        
        url = deck.df.loc[is_key, 'link'].tolist()[0]
        st.caption('Look it up on [woorden.org](%s)' % url)
        
        show_last_date = deck.df.loc[is_key, 'last_visit'].tolist()[0].date()
        show_last_score = deck.df.loc[is_key, 'last_score'].tolist()[0]
        st.caption(f'Previous score: {show_last_score}')
        st.caption(f'Previous review: {show_last_date}')
    with col3:
        get_slider_score(deck, key, session, i)

    sep_line()
    
    
def show_question_form(deck, practice_cards, session, q_count):      
    with st.form('form_questions'):

        for i in range(q_count):
            # The scores are stored in the session_state.q0 q1 ... etc. 
            show_one_question(deck, practice_cards, session, q_count, i)
        # Every form must have a submit button.
        # Run the function to update deck session state before the app is rerun
        # On click trigger the 'update' function to set 'st.session_state.submitted' to True
        st.form_submit_button(
            'Submit answers',
            on_click=update_submitted,
            args=(deck, practice_cards, session, q_count)
        )
            
def sep_line():
    st.write("""***""")
    
def create_deck_from_sample_data(data):
    # Import sample data
    word_list = data['string'].tolist()
    deck = deck_class.Deck(word_list)
    return deck

# st.image does not support gif yet. Below is a hack to do so, apart from linking a gif url.
# Credit: https://discuss.streamlit.io/t/image-in-markdown/13274/10
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

def update_submitted(deck, practice_cards, session, q_count):
    st.session_state.submitted = True
    st.session_state.deck_update_counter += 1
    if debug:
        st.warning(practice_cards)
        st.warning(f'q={q_count}')
        st.warning(f'session={session}')
    for i in range(q_count):
        # For each key (word) in the completed practice session
        # Update the user score to the dataframe
        key = practice_cards[i]
        score = st.session_state['q'+str(i)]
        deck.update_df(session, key, score) # Call deck class function to update stat in df

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')    


# external css

# callbacks

###############################################################################

###############################################################################
# Initialize
###############################################################################

# First we need a dataframe for containing the words and practice record
# We perform a check if there's already a dataframe in the session state
# The session state will be retained after a rerun (run the entire script again)
    
if 'submitted' not in st.session_state:
    st.session_state.submitted = None
    
if 'practice_cards' not in st.session_state:
    practice_cards = None
    st.session_state.practice_cards = practice_cards
    
if 'session' not in st.session_state:
    session = None
    st.session_state.session = session
    
if 'q_count' not in st.session_state:
    q_count = None
    st.session_state.q_count = q_count

# Load session state
practice_cards = st.session_state.practice_cards
session = st.session_state.session
q_count = st.session_state.q_count

###############################################################################
# Testing
###############################################################################
# Note to self: rmb the flow is linear and with rerun.
###############################################################################        
tab_h, tab_t, tab_w, tab_s, tab_r, tab_d = st.tabs(['Home', 'Test', 'Word list', 'Statistics', 'Reference', ':construction: Development'])
###############################################################################    
with st.sidebar:
   
    sep_line()
    
    with st.expander(':point_down: Choose a deck'):
        
        deck_options = {'Load deck': 'load', '1000 Dutch words (NOS)': 'new', '5 suits': 'simple_new'}
        deck_load_path = 'data/card_deck.csv'
        st.caption(f'Load path: {deck_load_path}')
        deck_mode = st.radio(
            label='Select deck to use',
            options=list(deck_options.keys()),
            key='deck_mode')

        # Initialize the update counter
        if 'deck_update_counter' not in st.session_state:
            st.session_state.deck_update_counter = 0
        # Initialize the new deck in session state
        if 'deck' not in st.session_state:
            # No need to run during reuns
            st.session_state.deck = None # None indicates no previous deck
        if st.button('Refresh deck'):
            st.session_state.deck = None # Reset such that deck2 is assigned again
        # Assign deck1 which will be used as the working deck
        if st.session_state.deck is None:
            # Use deck2 to denote a new deck, opposed to session state deck
            deck2 = create_new_deck(mode=deck_options[deck_mode], file_path=deck_load_path)
            deck1 = copy.deepcopy(deck2)
            st.session_state.deck = copy.deepcopy(deck2) # Also save to session state
        else:
            # A deck already exists
            deck1 = copy.deepcopy(st.session_state.deck) # Use the deck before rerun as working deck
    
    #sep_line()
    # Set up practice parameters
    with st.expander(':point_down: Configure practice session'):

        # Display input widgets for user to decide on q_count and session_select
        help_q_count = 'How many words would you like to practice in the session?'
        q_count_select = st.slider('How many questions?', min_value=1, max_value=20, step=1, help=help_q_count)

        help_session = 'An indexing/box system to rotate the sets of words for practice e.g. 0->1->...->9->0->...'
        box_options = [chr(x+97).upper() for x in range(10)]
        session_select = st.selectbox('Which box?', options=box_options, help=help_session)

        if st.button('Let\'s go :fire:', key='is_active_practice '):
            deck1, practice_cards, session, q_count = set_up_practice(deck1, q_count_select, session_select) # Set the parameters for practice
            # Update session state
            st.session_state.submitted = False # Reset submitted state when new practice begins
            st.session_state.deck = copy.deepcopy(deck1)
            st.session_state.practice_cards = practice_cards
            st.session_state.session = session
            st.session_state.q_count = q_count
        
    #sep_line()
    
    with st.expander(':point_down: Deck management'):
    
        # Download the dataframe as a CSV file
        st.download_button(
            label='Download current deck',
            data=convert_df(st.session_state.deck.df),
#            data=convert_df(deck1.df),
            file_name='card_deck.csv',
            mime='text/csv')
        
        # multiselect to filter data
        col_names = list(st.session_state.deck.df.columns)
        score_names = list(st.session_state.deck.df['n_scores'].iat[0].keys()) # s0...s5
        col_names = col_names + score_names
        cols_not_default = ['link', 'study_interval', 'last_visit', 'last_session', 'easiness', 'n_scores', 'box_id']  
        col_show_default = [x for x in col_names if x not in cols_not_default]
        selected_cols = st.multiselect('Show columns', col_names, col_show_default)
        
        score_names2 = [int(list(x)[1]) for x in score_names] # Convert 's0' to 0 etc.
        selected_scores = st.multiselect('Show words with last score', score_names2, score_names2)
        
        
    sep_line()
    

    st.sidebar.title('ik weet het niet')
    st.caption(
        """
        simply spaced repetition created by
        """)
    st.markdown(
        """
        :shark: | Freddy Feng
        """
    )
    st.caption(':book: | MSc Statistics & Data Science')
    st.markdown(
        """
         :speech_balloon: | [Profile](https://freddy-feng.github.io/postcard/) | [Linkedin](https://www.linkedin.com/in/freddy-feng/) | 
        [GitHub](https://github.com/freddy-feng/)
        """)
    st.caption(':smiley: Let me know what you think!')
    
    sep_line()

###############################################################################
with tab_h:
# First tab

    file_path = Path('gif/word_cloud_streamlit.gif')
    st.markdown(img_to_html(file_path), unsafe_allow_html=True) # Gif
    st.caption('1000 most frequent Dutch words extracted from news articles')

    
    with st.expander('About this app'):
        st.markdown(
            """
            - A toy project for personal use
            - An simple app for reinforcing one's memory
            - [Spaced repetition] trick (https://en.wikipedia.org/wiki/Spaced_repetition) in essence
                - Forget curve --> Recall --> Rinse and repeat
            - N.B. No idea how well it works... so give it a try
            """)
        
    with st.expander('Motivations'):
        st.markdown(
            """
            - Streamlit - what can it do?
            - Data analysis - mostly exploratory
            - Dutch - 1,000 words is just a starting point...
            - Memory training - does spaced repetition work for me?
            - App development - misschien?
            """)



###############################################################################    
with tab_t: # Test tab
# Display questions and submit answers
    st.subheader('Remember to press submit after answering all questions.')

    with st.expander('How to score'):
        st.markdown("""
            Use the slider to rate on how well you can answer a given card. A correct recall will remove it from the current deck and wait upon next review.
        
            :fire: __5__ - __Correct__ response with perfect recall.

            :stuck_out_tongue_winking_eye: __4__ - __Correct__ response, after some hesitation.

            :sweat: __3__ - __Correct__ response, but required significant effort to recall.

            :broken_heart: __2__ - __Wrong__ response, but upon seeing the correct answer it seemed easy to remember.

            :tired_face: __1__ - __Wrong__ response, but upon seeing the correct answer it felt familiar.

            :hankey: __0__ - __Wrong__ reponse. Total blackout!
        """)
    sep_line()
    
    # Answer not submitted = 
    if st.session_state.submitted is None:
        st.subheader('No question availble yet')
        st.write('Please set up a practice session from the __sidebar__ first. :point_left:')
    
    elif st.session_state.submitted:
        st.subheader(':thumbsup: Answer submitted')
        st.write(':point_left: Wanna start another session?')
        st.write(':point_right: Or look at the statistics?')
    
    elif (not st.session_state.submitted) & (q_count is not None):
        # User returns the scores and update the deck stat
        show_question_form(st.session_state.deck, practice_cards, session, q_count)
#        deck1 = show_question_form(deck1, practice_cards, session, q_count)
#        st.session_state.deck = copy.deepcopy(deck1) # Store to state
        
    else:
        raise ValueError('please take a look')
#------------------------------------------------------------------------------
with tab_w:
# Data description
    with st.expander('What is this word list about'):

        st.markdown("""
            - The word list is based on the most frequent words from the news articles published by NOS during 2010 - 2020.
            - It is prepared from counting the word frequency in the original texts.
            - Currently, the word list does not distinguish the words from verb conjugation, singular/plural form etc. Thus, it may be also correct to say that the word list consists of less than 1000 words.
            - The deck/word list can be downloaded in csv format by clicking the button at 'Deck Management' under the side bar.
            """)
    
    # Note to myself... better do a moving average?
    with st.expander('Does the quantity of news articles change over year?'):
        file_path = Path('fig/word_freq_heatmap.png')
        image = Image.open(file_path)
        st.image(image, caption='Monthly count of news articles in the dataset')
        
    with st.expander('What types of news articles are there?'):
        file_path = Path('fig/word_freq_categories.png')
        image = Image.open(file_path)
        st.image(image, caption='Categories of news articles in the dataset')
        
    with st.expander('How often does the 1000 most frequent words appear?'):
        file_path = Path('fig/word_freq_line.png')
        image = Image.open(file_path)
        st.image(image, caption='Distribution of frequent words in the dataset')

    with st.expander('Current deck content'):
        if st.session_state.deck is not None:
            st.write(st.session_state.deck.df['key'].tolist())
        
#------------------------------------------------------------------------------
    
with tab_s:
# Statistics page
    if st.session_state.deck is not None:
        df_show = st.session_state.deck.scores_dict_to_df(drop_scores_dict=False)


        # Plot score distribution
        with st.expander('Plot score counts'):
            
            df_scores = pd.DataFrame(df_show[score_names].sum(axis=0)).reset_index(drop=False) # Column sums for score counts
            df_scores = df_scores.rename(columns={'index': 'Score', 0: 'Counts'})
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(
                data=df_scores, x='Score', y='Counts'
            )
            plt.tight_layout()
            st.pyplot(fig)
        
 
        # Filter columns and rows
        with st.expander('Show deck data'):
            df_show = df_show.loc[df_show.last_score.isin(selected_scores), selected_cols]
            st.dataframe(df_show)
                
#        hide_cols = ['link', 'study_interval', 'previous_session', 'easiness']
#        st.dataframe(st.session_state.deck.scores_dict_to_df(drop_scores_dict=True).drop(hide_cols, axis=1))
#------------------------------------------------------------------------------
with tab_r:
# List references in the tab
    st.write(
        """
        - Thanks to Max Scheijen, the texts from NOS online news articles are scrapped and made available at   [https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles]. 
        """)
#------------------------------------------------------------------------------
with tab_d:
# For debug, development, displaying session state
    
    with st.expander('Session state:'):
        st.write(st.session_state)
    
    if debug:
        if st.session_state.deck is not None:
            st.write('Check: session_state.deck')
            st.dataframe(st.session_state.deck.df)           
            
    with st.expander('Development status'):
        st.write("""
        - This app is still under development.
        - Let me know if there's any issue!
        - Currently the spaced repetition is based on one of the [Leitner system](https://en.wikipedia.org/wiki/Leitner_system) variant.
        """)
    with st.expander('Future works'):
        st.write(""" 
        - Currently, the workflow involves manual labor: load->train->update->download->manually replace old file. This is something I want to improve.
        - Maybe a online host, fetching the csv from github or google sheet?
        - Plural, conjugations are still included in the list => actually less than 1000 words?
        - Deck
            - 1000 words from dutch lessons Bart de Pau, Dutchies to Be
        - Better and more useful functions!
            - Edit, add, remove words from deck
            - Experiment more interactive stuff in streamlit
            - It works like flashcard now, but it does not look like any flashcard now...
            - Showing the answer rather than the link to dictionary
            - Displaying statistics about the practice history and the word list
            - Interactive word cloud generator
            - Creating custom decks
        - Add an alternative mode based on [SM-2 algorithm](https://en.wikipedia.org/wiki/SuperMemo#Description_of_SM-2_algorithm). 
        - What's the difference between box-based and interval-based approach?
        - Some games inspired by Geheugentrainer (a Dutch TV program testing people's memory on grocery shopping)
        """)
#------------------------------------------------------------------------------

        
        
        
#------------------------------------------------------------------------------
# Future works 
#------------------------------------------------------------------------------
# Uploader 
# https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
# https://discuss.streamlit.io/t/uploading-csv-and-excel-files/10866/2
        
# proper cache
# https://discuss.streamlit.io/t/csv-dependency-on-github-deploy/26910/2
#------------------------------------------------------------------------------
           
#------------------------------------------------------------------------------
# Temp storage, not tested codes
#------------------------------------------------------------------------------
    # Upload a deck
#    uploaded_file = st.file_uploader("(BETA) Upload previous deck from local")
#    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
#        data = pd.read_csv(uploaded_file) # Previous df in csv
#        deck2 = deck_class.Deck() # Initialize
#        deck2.copy_df(data) # Load df and formatting
#        st.session_state.deck = deck2
#        deck = deck2
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Notes to myself
#------------------------------------------------------------------------------

# Also, when you change the selectbox, all of the functions inside the if st.button... will be rerun, so if any of them are slow, or intensive to run, you might want to use st.experimental_memo to cache them so they don’t need to be rerun every time.

# avoid using integer as dict key...

# Need more careful application on session state
#------------------------------------------------------------------------------


#############################
# Old codeblock initialize deck
#############################
# Initialize the counter in session state
#if 'deck_update_counter' not in st.session_state:
#    st.session_state.deck_update_counter = 0

# Initialize the new deck in session state
#if 'deck' not in st.session_state:
#    st.session_state.deck = None # None indicates no previous deck
    
# Create new deck i.e. deck 2
#if st.session_state.deck is None:
    # Use deck2 to denote a new deck, opposed to session state deck
#    deck2 = create_new_deck(mode='simple_new') # Using 5 suits
    #deck2 = create_new_deck(mode='new') # Using 1000 dutch words
    #deck2 = create_new_deck(mode='load') # Using previous deck
    
# Assign deck1 which will be used as the working deck
#if st.session_state.deck is None:
#    deck1 = copy.deepcopy(deck2) # Use the new deck2 as working deck
#else:
#    deck1 = copy.deepcopy(st.session_state.deck)     
#############################