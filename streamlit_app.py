###############################################################################
# Dependencies
import os
import random
from pathlib import Path
import copy

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64

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

def create_new_deck(is_deck2_create=False, is_deck2_load=True):   

    if is_deck2_create:
        # Create data for new user
        # Create a sample deck using 1000 dutch words
        data = fetch_sample_data()
        word_list = data['string'].tolist()
        deck2 = deck_class.Deck(word_list) # Create deck object

    if is_deck2_load:
        # Load deck csv from path
        csv_file_path = Path('data/card_deck.csv')
        data = pd.read_csv(csv_file_path) # Previous df in csv

        deck2 = deck_class.Deck() # Create deck object
        deck2.copy_df(data) # Load df and formatting
    
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
        st.write(f'Reduce number of questions to {q_max} because of too few current cards!')
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
    col1, col2, col3 = st.columns([2, 4, 3])
    with col1:
    # Display question number
        str_q = f"""<div style="text-align: center;"><br>{i+1} of {q_count}</div>"""
        st.markdown(str_q, unsafe_allow_html=True)
    with col2:
    # Display question
        st.header(f'__{key}__')
        is_key = deck.df['key'] == key
        url = deck.df.loc[is_key, 'link'].tolist()[0]
        st.write('Look it up on [woorden.org](%s).' % url)
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
        # painful
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



    
# Initialize the counter in session state
if 'deck_update_counter' not in st.session_state:
    st.session_state.deck_update_counter = 0

# Initialize the new deck in session state
if 'deck' not in st.session_state:
    st.session_state.deck = None # None indicates no previous deck
    
# Create new deck i.e. deck 2
if st.session_state.deck is None:
    # Use deck2 to denote a new deck, opposed to session state deck
    deck2 = create_new_deck()
    
# Assign deck1 which will be used as the working deck
if st.session_state.deck is None:
    deck1 = copy.deepcopy(deck2) # Use the new deck2 as working deck
else:
    deck1 = copy.deepcopy(st.session_state.deck) # Use the deck before rerun as working deck
    
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

practice_cards = st.session_state.practice_cards
session = st.session_state.session
q_count = st.session_state.q_count

###############################################################################
# Testing
###############################################################################
# Note to self: rmb the flow is linear and with rerun.
###############################################################################        
tab_h, tab_q, tab_w, tab_s, tab_r, tab_d = st.tabs(['Home', 'Questions', 'Word list', ':construction: Statistics', ':construction: Credits', ':construction: Development'])
###############################################################################    
with st.sidebar:
    sep_line()
    st.sidebar.title('ik weet het niet')
    st.caption(
        """
        simply spaced repetition...
        created by
        """)
    st.markdown(
        """
        :shark: | Freddy Feng
        """
    )
    st.caption(':book: | MSc Statistics & Data Science')
    st.markdown(
        """
         :speech_balloon: | [Linkedin](https://www.linkedin.com/in/freddy-feng/) | 
        [GitHub](https://github.com/freddy-feng/)
        """)
    
    sep_line()
    
    if debug:
        st.warning(f'session_state.deck is none = {st.session_state.deck is None}')
        st.warning(f'deck update count = {st.session_state.deck_update_counter}')
        st.warning(f'submitted state = {st.session_state.submitted}')

    
    st.header('Set up practice session here :point_down:')

    with st.expander('How to use'):
        st.markdown(
            """
            1. Set up a practice session under the side bar
            2. Answer questions under the Questions tab
            3. Submit answers
            4. Rinse and repeat
            """)
    
    # Display input widgets for user to decide on q_count and session_select
    help_q_count = 'How many words would you like to practice in the session?'
    q_count_select = st.slider('Select questions to practice', min_value=1, max_value=20, step=1, help=help_q_count)
    
    help_session = 'An indexing/box system to rotate the sets of words for practice e.g. 0->1->...->9->0->...'
    session_select = st.selectbox('Select index to practice', options=np.arange(10), help=help_session)
    
    if st.button('Let\'s go :fire:', key='is_active_practice '):
        deck1, practice_cards, session, q_count = set_up_practice(deck1, q_count_select, session_select) # Set the parameters for practice
        # Update session state
        st.session_state.submitted = False # Reset submitted state when new practice begins
        st.session_state.deck = copy.deepcopy(deck1)
        st.session_state.practice_cards = practice_cards
        st.session_state.session = session
        st.session_state.q_count = q_count
        
    sep_line()
    
    st.header('Deck managment')
    
    # Download the dataframe as a CSV file
    st.download_button(
        label='Download current deck',
        data=convert_df(deck1.df),
        file_name='card_deck.csv',
        mime='text/csv')
    st.warning(
        """
        To be added:
        - Upload a csv i.e. list or previous deck
        - Reset deck
        """)
    
    sep_line()

###############################################################################
with tab_h:
# First tab

    file_path = Path('gif/word_cloud_streamlit.gif')
    st.markdown(img_to_html(file_path), unsafe_allow_html=True) # Gif
    st.caption('1000 most frequent Dutch words from news articles')

    
    with st.expander('What is this app'):
        st.markdown(
            """
            - A toy project for personal use
            - [Spaced repetition] trick (https://en.wikipedia.org/wiki/Spaced_repetition) in essence
                - Forget curve --> Recall --> Rinse and repeat
                - No idea how well it works... so give it a try""")
        
    with st.expander('Motivations'):
        st.markdown(
            """
            - Streamlit - what can it do?
            - Data analysis - mostly exploratory
            - Dutch - 1,000 words is just a starting point...
            - Memory training - does spaced repetition work for me?
            - App development - misschien?
            """)

    sep_line()    
    st.write('Let me know what you think :smiley:')

###############################################################################    
with tab_q:
# Display questions and submit answers
    st.header('How well do you remember the word?')
    st.write('Rate from 0 (nope) to 5 (perfect), where 3 or above is a successful recall.')

    with st.expander('What does the score stand for'):
        st.markdown("""
            :fire: 5: Correct response with perfect recall.

            :stuck_out_tongue_winking_eye: 4: Correct response, after some hesitation.

            :sweat: 3: Correct response, but required significant effort to recall.

            :broken_heart: 2: Incorrect response, but upon seeing the correct answer it seemed easy to remember.

            :tired_face: 1: Incorrect response, but upon seeing the correct answer it felt familiar.

            :hankey: 0: Total blackout!
        """)
    sep_line()
    
    # Answer not submitted = 
    if st.session_state.submitted is None:
        st.subheader('No question availble yet')
        st.write('Please set up a practice session from the __sidebar__ first. :point_left:')
    
    elif st.session_state.submitted:
        st.subheader('Answer submitted :thumbsup:')
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
            - Thanks to Max Scheijen, the texts from NOS online news articles are scrapped and made available at   [https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles]. 
            - The word list is then prepared by counting the occurence and sorting, after tokenizing and wrangling.
            - The deck/word list can be downloaded in csv format by clicking the button under the side bar.
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

    if st.session_state.deck is not None:
        st.write('Show sessions state deck')
        st.write(st.session_state.deck.df)
        
#------------------------------------------------------------------------------
    
with tab_s:
# Statistics page
    st.write('Under construction')
    st.write('Show score')
    st.write('Show easiness')
    st.write('Show deck1')
    st.write(deck1.df)
    st.write(deck1.df['key'].tolist())
#------------------------------------------------------------------------------
with tab_r:
# List references in the tab
    st.write('Under construction')
#------------------------------------------------------------------------------
with tab_d:
# For debug, development, displaying session state
    if debug:
        st.write(st.session_state)
        
    with st.expander('Development status'):
        st.write("""
        - This app is still under development.
        - Let me know if there's any issue!
        - Currently the spaced repetition is based on one of the [Leitner system](https://en.wikipedia.org/wiki/Leitner_system) variant.
        """)
    with st.expander('Future works'):
        st.write(""" 
        - Better and more useful functions!
            - Experiment more interactive stuff in streamlit
            - It works like flashcard now, but it does not look like any flashcard now...
            - Showing the answer rather than the link to dictionary
            - Displaying statistics about the practice history and the word list
            - Interactive word cloud generator
            - Creating custom decks
        - Add an alternative mode based on [SM-2 algorithm](https://en.wikipedia.org/wiki/SuperMemo#Description_of_SM-2_algorithm). 
        - What's the difference between box-based and interval-based approach?
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
# Temp storage
#------------------------------------------------------------------------------
            # Add the score to the corresponding count in the dataframe
#            idx = df.index[df['suit'] == suit].tolist()[0]
#            df.loc[idx, 'count'] += score

#    elif st.button('Create deck'):
        # Download the dataframe as a CSV file
#        st.download_button(
#            label='Upload a CSV file consisting a list of things you want to memorize',
#            data=df.to_csv(index=False),
#            file_name='card_deck.csv',
#            mime='text/csv'
#        )
#------------------------------------------------------------------------------
#    if st.button('(BETA) Reset deck'):
#        del st.session_state['deck_update_counter']
#        del st.session_state['deck']
    
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
