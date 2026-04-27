import pandas as pd
import random

# Define categories and their Hausa keyword/phrase templates
templates = {
    "conflict": [
        "An kai hari a garin {town} jiya da daddare.",
        "'Yan bindiga sun sace mutane {count} a hanyar {town}.",
        "Rikici ya barke tsakanin {group1} da {group2} a {town}.",
        "An ji karar harbe-harbe a yankin {town} tun safe.",
        "Sojoji sun fatattaki 'yan ta'adda a dajin {town}.",
        "An kona gidaje da shaguna a rikicin {town}.",
        "Harin kunar bakin wake ya faru a {town}.",
        "Mutane da dama sun rasa rayukansu a sabon harin {town}.",
        "An tsinci gawarwakin mutane bayan harin 'yan bindiga.",
        "An sanya dokar hana fita a {town} saboda rashin tsaro."
    ],
    "displacement": [
        "Dubban mutane sun gudu daga {town} zuwa sansanin gudun hijira.",
        "Muna bukatar matsuguni saboda an kore mu daga gidajenmu a {town}.",
        "Iyalanmu suna kwana a waje saboda rashin gida.",
        "Mutane sun bar {town} saboda tsoron hare-hare.",
        "Sansanin IDP na {town} ya cika da mutane.",
        "Mun rasa komai namu, muna gudun hijira yanzu.",
        "Yara da mata suna wahala a sansanin gudun hijira na {town}.",
        "An tilasta wa mazauna {town} barin garinsu.",
        "Ba mu da wurin kwana, muna bukatar taimako a {town}.",
        "Muna neman mafaka a makarantar {town}."
    ],
    "food_insecurity": [
        "Yunwa tana kashe mutane a {town}.",
        "Babu abinci a wannan yankin kwata-kwata.",
        "Farashin hatsi ya tashi a {town}, talaka ba zai iya saya ba.",
        "Yara suna fama da rashin abinci mai gina jiki a {town}.",
        "Muna bukatar agajin abinci cikin gaggawa a {town}.",
        "Amfanin gona ya lalace a {town}, babu abin da za mu ci.",
        "Mutane suna cin ganyen bishiya saboda yunwa.",
        "Ba mu ci abinci ba tun jiya a {town}.",
        "Rashin abinci ya sa mutane da yawa rashin lafiya.",
        "Agajin abinci da aka kawo {town} bai isa kowa ba."
    ],
    "disease_outbreak": [
        "Cutar {disease} tana bazuwa a garin {town}.",
        "Asibitin {town} ya cika da masu amai da gudawa.",
        "Mutane da yawa suna mutuwa a {town} saboda rashin magani.",
        "An gano sabon nau'in cuta a yankin {town}.",
        "Hukumomin lafiya sun ba da gargadi game da cutar {disease} a {town}.",
        "Yara suna bukatar rigakafin cutar {disease} a {town}.",
        "Cutar ta shafi kauyuka da dama a karamar hukumar {town}.",
        "Babu isassun ma'aikatan lafiya don kula da marasa lafiya a {town}.",
        "Muna bukatar magunguna da kayan aiki a asibitin {town}.",
        "An killace wasu mutane saboda zargin cutar {disease}."
    ],
    "flood": [
        "Ambaliyar ruwa ta mamaye garin {town}.",
        "Ruwan sama ya rushe gidaje sama da dari a {town}.",
        "Gonaki sun nutse cikin ruwa a {town}, manoma sun yi asara.",
        "An katse hanyar {town} saboda ambaliya.",
        "Mutane suna amfani da kwalara don tsallakawa a {town}.",
        "Ruwa ya tafi da dabbobinmu da kayan amfanin gona a {town}.",
        "Muna bukatar taimakon gaggawa bayan ambaliyar {town}.",
        "Ambaliyar ta shafi unguwanni da dama a cikin birnin {town}.",
        "Ruwa ya shiga cikin dakunanmu a {town}, ba mu da wurin kwana.",
        "An ba da sanarwar ambaliya ga mazauna gabar kogi a {town}."
    ],
    "no_crisis": [
        "An yi biki lafiya a garin {town}.",
        "Kasuwar {town} ta cika da mutane yau.",
        "Gwamnati ta kaddamar da sabon aikin hanya a {town}.",
        "Dalibai sun koma makaranta a {town} bayan hutu.",
        "Manoma suna girbin amfanin gona a {town} wannan kakar.",
        "An gudanar da wasan kwallon kafa a filin wasa na {town}.",
        "Garin {town} yana cikin kwanciyar hankali da zaman lafiya.",
        "An bude sabon babban kanti a tsakiyar gari na {town}.",
        "Jama'a suna gudanar da harkokinsu na yau da kullum a {town}.",
        "Yanayi yana da kyau sosai a yau a {town}."
    ]
}

towns = ["Maiduguri", "Kano", "Kaduna", "Katsina", "Sokoto", "Zaria", "Bauchi", "Gombe", "Damaturu", "Jalingo", "Potiskum", "Hadejia", "Daura", "Bichi", "Gumel", "Yola", "Dutse", "Lafia", "Minna", "Gusau"]
groups = ["manoma", "makiyaya", "sojoji", "'yan bindiga", "matasa", "mazauna gari", "yan banga"]
diseases = ["kwalara", "sankarau", "polio", "zazzabin lassa", "korona", "kyanda"]

expanded_data = []

# Generate 30 unique examples per category
for label, template_list in templates.items():
    for _ in range(30):
        template = random.choice(template_list)
        text = template.format(
            town=random.choice(towns),
            count=random.randint(5, 50),
            group1=random.choice(groups),
            group2=random.choice(groups),
            disease=random.choice(diseases)
        )
        expanded_data.append({"text": text, "label": label})

# Load original data
original_df = pd.read_csv("/home/ubuntu/hausa-project/hausa_crisis_data.csv")
new_df = pd.DataFrame(expanded_data)

# Combine
final_df = pd.concat([original_df, new_df], ignore_index=True)

# Save to the correct folder structure as requested
import os
os.makedirs("/home/ubuntu/hausa-project/data", exist_ok=True)
final_df.to_csv("/home/ubuntu/hausa-project/data/hausa_crisis_data.csv", index=False)

print(f"Dataset expanded. Total rows: {len(final_df)}")
print(final_df['label'].value_counts())
