import streamlit as st

def main():
    st.title('A Deep Learning based approach to detecting forest fires')

    # Display "About Me" section
    if st.button('About Me'):
        st.title('About Me')
        st.write("Hello! I am Sarrang, the creator of this Deep Learning based forest fire detection system. I am passionate about leveraging technology to address environmental challenges. If you have any questions or would like to connect, feel free to reach out to me.")
        st.image("your_profile_picture.jpg", use_column_width=True, caption="Sarang")
        st.write("Email: your-email@example.com")
        st.write("Social Media: [LinkedIn](https://www.linkedin.com/in/b-sarang-8b5b20217/), [Twitter](https://www.twitter.com/your-twitter-profile)")

    # Rest of your code

if __name__ == '__main__':
    main()
