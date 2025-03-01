import streamlit as st

st.markdown("""
    <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 2rem 0;
            text-align: center;
        }
        
        /* Improved text styling */
        .full-width-text {
            text-align: justify;
            font-size: 18px;
            line-height: 1.6;
            margin-bottom: 2rem;
        }
        
        /* Container for better alignment */
        .content-container {
            display: flex;
            align-items: center;
            gap: 0;
            margin: 1rem 0;
        }
        
        /* Right-aligned image container */
        .image-right {
            display: flex;
            justify-content: flex-end;
            margin-left: auto;
            padding: 0;
            width: 100%;
        }
        
        /* Left-aligned image container */
        .image-left {
            display: flex;
            justify-content: flex-start;
            margin-right: auto;
            padding: 0;
            width: 100%;
        }
        
        /* Text content styling */
        .text-content {
            flex: 1;
            padding-right: 2rem;
        }
        
        /* Section divider */
        .section-divider {
            margin: 2rem 0;
            border-top: 1px solid #e5e5e5;
        }
        
        /* Heading styles */
        h1, h2, h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        /* Link styling */
        a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Image styling */
        img {
            width: 100%;
            object-fit: cover;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Page title and banner
    st.title("🏥 AI in Healthcare: Optimizing Resources & Accessibility")
    st.image("bgImage/healthcare_banner.jpg", use_container_width=True)

    # Introduction section
    st.markdown("""
        <div class="full-width-text">
            As Singapore's healthcare system develops and continues to evolve throughout the years, leveraging cutting-edge technologies 
            becomes an important factor to address emerging challenges in Singapore. As the demand for medical services and resources continues to rise due to  epidemiological 
            transition, traditional approaches may no longer be enough to resolve these issues. However, Artificial Intelligence (AI), may be able to 
            provide help for our healthcare sector by offering them the potential to enhance efficiency in hospitals, optimise healthcare resources, and 
            improve accessibility. Hence, these possibilities lead us to form our problem statement: 
            <b>How can AI help optimize healthcare resources and improve accessibility for Singaporeans?</b>
        </div><br><br>
    """, unsafe_allow_html=True)

    # Page title with larger font
    st.markdown('<h1 class="main-title">🔍 How AI is Transforming Healthcare in Singapore</h1><br><br>', unsafe_allow_html=True)

    # Content sections using custom columns with adjusted image sizes
    # 1. Disease Prediction
    col1, col2 = st.columns([2, 1], gap="small")
    with col1:
        st.markdown("""
            ### 1️⃣ Disease Prediction System
            AI models can analyze patient symptoms, genetic data, and medical history to **predict diseases early**.
            By identifying patterns such as symptom to disease and symptom to other symptoms, AI helps doctors make informed decisions and improve **early diagnosis rates**.
            In addition, it will be sustainable for the healthcare sector with regards to **Manpower**, where healthcare sectors can delegate doctors to more important sectors of healthcare.
            👉 [Learn more](https://www.youtube.com/watch?v=P5jTcEWhJMc)
        """)
    with col2:
        st.image("bgImage/disease_prediction.jpeg", use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 2. Medicine Image Classification
    col3, col4 = st.columns([1, 2], gap="small")
    with col3:
        st.image("bgImage/image_classification.jpg", use_container_width=True)
    with col4:
        st.markdown("""
            ### 2️⃣ Medicine Image Classification
            AI-powered image classification enables pharmacies and hospitals to accurately identify medications, decreasing 
            Look-Alike Sound-Alike (LASA) drug mistakes. By freeing up workers to provide direct patient care, it improves patient 
            safety, reduces the need for manual checks, and maximizes healthcare resources. It speeds up prescription processing, simplifies 
            operations, and lowers the possibility of human error by automating repetitive tasks. It also ensures that the right 
            drug is given by preventing pharmaceutical mix-ups. AI is a useful support tool for medication administration and the provision 
            of healthcare overall because, although it increases efficiency, human oversight guarantees dependability.
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 3. Predicting Outpatient Attendance
    col5, col6 = st.columns([2, 1], gap="small")
    with col5:
        st.markdown("""
            ### 3️⃣ Predicting Outpatient Attendance
            AI can be used to help predict **outpatient attendance** by analysing past patient visits in public healthcare facilities. 
            It can also provide healthcare personnel recommendation based on the predicted outpatient attendance which will
            help our public healthcare facilities to better align staffing needs with patient demand, reducing both over and under staffing scenarios. 
            Therefore, this enhances operational efficiency, minimises resource waste, and optimises healthcare resources, contributing to sustainable healthcare practices.
        """)
    with col6:
        st.image("bgImage/outpatient.jpg", use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 4. Hospital Capacity Prediction
    col7, col8 = st.columns([1, 2], gap="small")
    with col7:
        st.image("bgImage/bed_occupancy.jpeg", use_container_width=True)
    with col8:
        st.markdown("""
            ### 4️⃣ Hospital Capacity Prediction
            AI-driven models analyze hospital data to forecast bed demand and optimize hospital occupancy rates. This prevents overcrowding, 
            improves emergency care, and ensures better resource management. By identifying capacity trends in advance, 
            hospitals can allocate resources more effectively, align staffing levels with patient demand, and enhance overall patient care. 
            This proactive approach minimizes inefficiencies and supports a sustainable, efficient healthcare system.
        """)

if __name__ == "__main__":
    main()
