import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

h1{
    color:#4CAF50;
}

h2,h3{
    color:white;
}

[data-testid="stMetricValue"]{
    color:#4CAF50;
}

[data-testid="stMetricLabel"]{
    color:white;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:3em;
    background:#4CAF50;
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🤖 AI Career FAQ Chatbot")

st.sidebar.info("""
### Horizon TechX Internship

**Task 2 – AI Chatbot for FAQs**

### Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- NLP (TF-IDF)

### Features

✅ FAQ Chatbot

✅ Similarity Matching

✅ Career Guidance

✅ Skill Gap Analysis

✅ Learning Resources

✅ Career Explorer

✅ Analytics

""")

# --------------------------------------------------
# FAQ DATASET
# --------------------------------------------------

faq_data = {

"Question":[

"How do I become a Data Scientist?",

"What skills are required for AI Engineer?",

"How can I become a Software Developer?",

"What skills are needed for Cloud Engineer?",

"What certifications should I do?",

"Is Python necessary for AI?",

"What is the salary of a Data Analyst?",

"Difference between Data Scientist and Data Analyst?",

"How do I start Machine Learning?",

"What roadmap should I follow for Full Stack Development?"

],

"Answer":[

"Learn Python, Statistics, Machine Learning, SQL and build projects.",

"Learn Python, Deep Learning, NLP, TensorFlow and build AI projects.",

"Learn Programming, DSA, DBMS and build software projects.",

"Learn Linux, AWS, Docker and Cloud Computing fundamentals.",

"Python for Everybody, AWS Cloud Practitioner and Machine Learning Specialization are excellent choices.",

"Yes. Python is the most widely used programming language for AI and Machine Learning.",

"The average salary ranges between 5-12 LPA depending on experience and company.",

"Data Scientists build predictive models while Data Analysts mainly analyze business data.",

"Start with Python, Statistics, Machine Learning algorithms and hands-on projects.",

"Learn HTML, CSS, JavaScript, React, Node.js and databases."

]

}

faq_df = pd.DataFrame(faq_data)
# --------------------------------------------------
# CAREER DATASET
# --------------------------------------------------

career_data = {

"Career":[
"Data Scientist",
"AI Engineer",
"Data Analyst",
"Software Developer",
"Cloud Engineer",
"Machine Learning Engineer",
"Frontend Developer",
"Backend Developer",
"Full Stack Developer",
"Cyber Security Analyst",
"DevOps Engineer",
"Database Administrator"
],

"Description":[
"python machine learning statistics sql analytics",
"python deep learning tensorflow artificial intelligence",
"excel sql power bi dashboards analytics",
"java dsa programming software development",
"aws docker linux cloud computing",
"python machine learning tensorflow ai",
"html css javascript react",
"python java api database backend",
"html css javascript react nodejs",
"networking linux cyber security",
"docker kubernetes aws linux",
"sql mysql postgresql database"
],

"Skills":[
"Python, Machine Learning, Statistics",
"Python, Deep Learning, TensorFlow",
"Excel, SQL, Power BI",
"Java, DSA, DBMS",
"AWS, Docker, Linux",
"Python, TensorFlow, Machine Learning",
"HTML, CSS, JavaScript, React",
"Python, Java, SQL",
"HTML, CSS, JavaScript, React, Node.js",
"Networking, Linux, Security",
"Docker, Kubernetes, AWS",
"SQL, MySQL, PostgreSQL"
],

"Roadmap":[
"Python → Statistics → ML → Projects",
"Python → DL → NLP → AI Projects",
"Excel → SQL → Power BI",
"Programming → DSA → Projects",
"Linux → AWS → Docker",
"Python → ML → TensorFlow",
"HTML → CSS → JavaScript → React",
"Programming → APIs → Databases",
"HTML → CSS → JavaScript → React → Node.js",
"Networking → Linux → Security",
"Docker → Kubernetes → AWS",
"SQL → Databases → Administration"
],

"Salary":[
"8-20 LPA",
"10-25 LPA",
"5-12 LPA",
"6-18 LPA",
"7-18 LPA",
"10-22 LPA",
"5-15 LPA",
"6-18 LPA",
"6-18 LPA",
"6-15 LPA",
"8-20 LPA",
"6-16 LPA"
]

}

career_df = pd.DataFrame(career_data)

# --------------------------------------------------
# HOME
# --------------------------------------------------

st.title("🤖 AI Career FAQ & Guidance Chatbot")

st.write("""
Welcome to the **AI Career FAQ Chatbot**.

This application uses **Natural Language Processing (NLP)** and
**TF-IDF Similarity Matching** to answer career-related questions,
recommend suitable career paths, identify skill gaps and provide
learning recommendations.
""")

st.markdown("---")

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("FAQ Questions","10")

with col2:
    st.metric("Career Paths","12")

with col3:
    st.metric("AI Engine","TF-IDF")

with col4:
    st.metric("Interface","Interactive")

st.markdown("---")
# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

st.header("💬 Ask the AI Career Assistant")

st.write("Ask any career-related question or describe your interests.")

question = st.text_input(
    "💭 Your Question",
    placeholder="Example: How do I become a Data Scientist?"
)

st.markdown("### OR")

name = st.text_input(
    "👤 Your Name"
)

degree = st.selectbox(
    "🎓 Select Your Degree",
    [
        "BCA",
        "B.Tech",
        "MCA",
        "B.Sc",
        "B.E",
        "Other"
    ]
)

interest = st.text_input(
    "💡 Your Skills / Interests",
    placeholder="Example: Python, SQL, Machine Learning"
)

st.markdown("")

if st.button("🤖 Ask Career Chatbot"):

    if question.strip() == "" and interest.strip() == "":
        st.warning("Please enter a question or your interests.")
        st.stop()

    # -----------------------------
    # FAQ Similarity Matching
    # -----------------------------

    documents = list(faq_df["Question"])

    if question.strip() != "":
        user_input = question.lower()
    else:
        user_input = interest.lower()

    documents.append(user_input)

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    best_index = similarity.argmax()

    best_score = similarity[best_index] * 100

    st.markdown("---")

    st.subheader("🤖 AI Chatbot Response")

    st.success(
        f"""
Hello **{name if name else 'User'}** 👋

I analyzed your question using **Natural Language Processing (TF-IDF Similarity Matching).**

### Closest FAQ Match

**{faq_df.iloc[best_index]['Question']}**

### Answer

{faq_df.iloc[best_index]['Answer']}
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Similarity Score",
            f"{best_score:.1f}%"
        )

    with col2:
        st.metric(
            "FAQ Database",
            f"{len(faq_df)} Questions"
        )

    st.markdown("---")
# --------------------------------------------------
# AI CAREER MATCHING
# --------------------------------------------------

if interest.strip() != "":

    # Career Similarity
    documents = list(career_df["Description"])
    documents.append(interest.lower())

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    career_df["Match"] = similarity * 100

    top_careers = career_df.sort_values(
        by="Match",
        ascending=False
    ).head(3)

    top = top_careers.iloc[0]
    other = top_careers.iloc[1:]

    st.header("🎯 AI Career Recommendations")

    st.write(
        "Based on your interests, these careers best match your profile."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🥇 Best Career", top["Career"])

    with c2:
        st.metric("📈 Match", f"{top['Match']:.1f}%")

    with c3:
        st.metric("💰 Salary", top["Salary"])

    st.markdown("---")

    st.subheader("🥇 Best Career Recommendation")

    with st.container(border=True):

        st.subheader(f"💼 {top['Career']}")

        st.progress(min(int(top["Match"]),100))

        st.write("### 💰 Salary")
        st.write(top["Salary"])

        st.write("### 🛠 Required Skills")

        skills = [s.strip() for s in top["Skills"].split(",")]

        for skill in skills:
            st.write(f"✅ {skill}")

        st.write("### 🗺 Learning Roadmap")

        roadmap = [r.strip() for r in top["Roadmap"].split("→")]

        for step in roadmap:
            st.write(f"➡️ {step}")

        # Skill Gap

        st.write("### 📊 Skill Gap Analysis")

        user_skills = [
            x.strip().lower()
            for x in interest.split(",")
        ]

        missing_skills = []

        for skill in skills:

            if skill.lower() not in user_skills:
                missing_skills.append(skill)

        if len(missing_skills)==0:

            st.success(
                "🎉 You already have all required skills."
            )

        else:

            for skill in missing_skills:
                st.write(f"❌ {skill}")
        # ------------------------------------------
        # Learning Resources
        # ------------------------------------------

        learning_resources = {
            "Python":"Python for Everybody (Coursera)",
            "Machine Learning":"Machine Learning Specialization",
            "Statistics":"Khan Academy Statistics",
            "TensorFlow":"TensorFlow Developer Course",
            "Deep Learning":"Deep Learning Specialization",
            "SQL":"SQLBolt",
            "MySQL":"MySQL Official Tutorial",
            "PostgreSQL":"PostgreSQL Official Documentation",
            "Java":"Java Programming Masterclass",
            "DSA":"GeeksforGeeks DSA",
            "DBMS":"DBMS Complete Course",
            "Power BI":"Microsoft Learn",
            "AWS":"AWS Cloud Practitioner",
            "Docker":"Docker for Beginners",
            "Linux":"Linux Journey",
            "React":"React Complete Guide",
            "Networking":"Cisco Networking Basics",
            "Security":"Google Cybersecurity Certificate"
        }

        if missing_skills:

            st.write("### 📚 Recommended Learning Resources")

            for skill in missing_skills:

                if skill in learning_resources:
                    st.write(f"📖 **{skill}** → {learning_resources[skill]}")

        # ------------------------------------------
        # Certifications
        # ------------------------------------------

        certifications = {
            "Python":"Python for Everybody",
            "Machine Learning":"Machine Learning Specialization",
            "TensorFlow":"TensorFlow Developer",
            "AWS":"AWS Cloud Practitioner",
            "SQL":"Oracle SQL Associate",
            "Java":"Oracle Java Associate",
            "Docker":"Docker Certified Associate",
            "Linux":"Linux Essentials",
            "React":"Meta Front-End Developer",
            "MySQL":"MySQL Database Administrator",
            "PostgreSQL":"PostgreSQL Associate"
        }

        certs = [s for s in missing_skills if s in certifications]

        if certs:

            st.write("### 🏅 Recommended Certifications")

            for skill in certs:
                st.write(f"🏆 {certifications[skill]}")

        # ------------------------------------------
        # Resume Tips
        # ------------------------------------------

        resume_tips = {
            "Data Scientist":[
                "Build Machine Learning Projects",
                "Create a Kaggle Profile",
                "Show Python & SQL skills"
            ],
            "AI Engineer":[
                "Build AI Chatbots",
                "Show TensorFlow Projects",
                "Include Deep Learning models"
            ],
            "Data Analyst":[
                "Create Power BI Dashboards",
                "Highlight SQL Skills",
                "Show Excel Projects"
            ],
            "Software Developer":[
                "Build Full Stack Projects",
                "Strong GitHub Portfolio",
                "Practice DSA"
            ]
        }

        if top["Career"] in resume_tips:

            st.write("### 📄 Resume Recommendations")

            for tip in resume_tips[top["Career"]]:
                st.write(f"✅ {tip}")

        # ------------------------------------------
        # 6 Month Plan
        # ------------------------------------------

        st.write("### 🗓 Personalized 6-Month Plan")

        months = [
            "Month 1","Month 2","Month 3",
            "Month 4","Month 5","Month 6"
        ]

        for i, month in enumerate(months):

            if i < len(missing_skills):
                st.write(f"📘 **{month}:** Learn {missing_skills[i]}")
            elif i == len(missing_skills):
                st.write(f"💻 **{month}:** Build Projects")
            elif i == len(missing_skills)+1:
                st.write(f"🏅 **{month}:** Earn Certifications")
            else:
                st.write(f"🚀 **{month}:** Apply for Internships")

    # ------------------------------------------
    # Other Career Recommendations
    # ------------------------------------------

    st.markdown("---")
    st.header("📂 Other Career Recommendations")

    for _, row in other.iterrows():

        with st.expander(f"💼 {row['Career']} ({row['Match']:.1f}% Match)"):

            st.write(f"**💰 Salary:** {row['Salary']}")
            st.write(f"**🛠 Skills:** {row['Skills']}")
            st.write(f"**🗺 Roadmap:** {row['Roadmap']}")

    # ------------------------------------------
    # Comparison Chart
    # ------------------------------------------

    st.markdown("---")
    st.header("📊 Career Comparison")

    fig = px.bar(
        top_careers,
        x="Career",
        y="Match",
        color="Career",
        text="Match"
    )

    fig.update_traces(texttemplate="%{text:.1f}%")

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # Career Explorer
    # ------------------------------------------

    st.markdown("---")
    st.header("🧭 Explore All Careers")

    selected = st.selectbox(
        "Select a Career",
        career_df["Career"]
    )

    info = career_df[career_df["Career"] == selected].iloc[0]

    with st.container(border=True):

        st.subheader(info["Career"])
        st.write(f"**💰 Salary:** {info['Salary']}")
        st.write(f"**🛠 Skills:** {info['Skills']}")
        st.write(f"**🗺 Roadmap:** {info['Roadmap']}")

    # ------------------------------------------
    # Download Report
    # ------------------------------------------

    report = f"""
AI Career Guidance Report

Name: {name}
Degree: {degree}

Best Career:
{top['Career']}

Match Score:
{top['Match']:.1f}%

Salary:
{top['Salary']}

Skills:
{top['Skills']}

Roadmap:
{top['Roadmap']}
"""

    st.download_button(
        "📄 Download Career Report",
        report,
        "career_report.txt",
        "text/plain"
    )