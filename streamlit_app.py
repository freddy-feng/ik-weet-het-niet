import streamlit as st
import pandas as pd
import random

# Define the initial pandas dataframe with suits and counts
data = {
    'suit': ['Hearts', 'Diamonds', 'Clubs', 'Spades'],
    'count': [0, 0, 0, 0]
}
df = pd.DataFrame(data)

# Define the Streamlit app interface
def app():
    st.title('Card Practice')
    st.write('Welcome to Card Practice! Select an option below to get started.')
    
    # Create the 'Start practice' button and the 'Download deck' button
    if st.button('Start practice'):
        # Prompt user for number of questions
        q_count = st.number_input('Your dedication to this practice?', min_value=1, max_value=100, step=1)
        
        # Display a suit randomly drawn from the dataframe
        suit = random.choice(df['suit'])
        st.write(f"The drawn suit is: {suit}")
        
        # Create score buttons and ask the user a question
        for i in range(q_count):
            st.write(f"Question {i+1}: Do you know the card in {suit}?")
            score = st.button('1') + st.button('2')*2 + st.button('3')*3 + st.button('4')*4 + st.button('5')*5
            
            # Add the score to the corresponding count in the dataframe
            idx = df.index[df['suit'] == suit].tolist()[0]
            df.loc[idx, 'count'] += score
            
            # Choose the next suit randomly from the dataframe
            suit = random.choice(df['suit'])
            st.write(f"The drawn suit is: {suit}")
        
        # Display a message when the practice is completed
        st.write('Hoorah! See you in next practice :)')
    elif st.button('Download deck'):
        # Download the dataframe as a CSV file
        st.download_button(
            label="Download the deck as a CSV file",
            data=df.to_csv(index=False),
            file_name='card_deck.csv',
            mime='text/csv'
        )