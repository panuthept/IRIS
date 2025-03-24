from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


things_not_to_do = """### **General Etiquette & Behavior**  
1. Touching someone’s head (considered the most sacred part of the body).  
2. Pointing at people with your feet (feet are considered the lowest part of the body).  
3. Touching Buddhist monks, especially if you are a woman.  
4. Wearing shoes inside temples or homes.  
5. Stepping over books or money (disrespectful, as books symbolize knowledge and money often has sacred images).  
6. Hugging or kissing in public.  
7. Speaking loudly or arguing in public.  
8. Using your left hand to give or receive something (considered unclean).  
9. Whistling at night (thought to attract evil spirits).  
10. Sitting higher than a monk in any setting.  
11. Walking directly in front of someone praying.  
12. Sitting with your feet stretched toward a Buddha statue.  
13. Touching or pointing at Buddha images or religious statues.  
14. Disrespecting elders by not addressing them properly.  
15. Not using honorifics when speaking to older people.  
16. Failing to remove your hat or sunglasses when speaking to elders.  
17. Showing excessive displays of wealth or boasting.  
18. Speaking negatively about Buddhism or religious beliefs.  
19. Eating before elders have started their meal.  
20. Leaning on or resting against temple structures.  

### **Dress Code & Appearance**  
21. Wearing revealing clothing at temples (shorts, sleeveless tops, etc.).  
22. Wearing black to a happy event (associated with funerals).  
23. Showing tattoos of Buddha or religious figures (considered highly offensive).  
24. Wearing shoes with religious symbols.  
25. Going shirtless in public (for men).  
26. Wearing a mini skirt or tight clothes in traditional settings.  
27. Wearing inappropriate clothing to religious festivals.  
28. Failing to cover your knees and shoulders in rural areas.  
29. Wearing shoes inside someone’s home without being invited to.  
30. Not dressing modestly when visiting elderly people.  

### **Religious Practices & Beliefs**  
31. Touching Buddhist monks without permission.  
32. Taking selfies with monks without their consent.  
33. Disrespecting alms-giving traditions.  
34. Entering a pagoda without removing socks.  
35. Climbing on or sitting on a stupa.  
36. Talking during religious ceremonies.  
37. Eating inside a temple.  
38. Offering alcohol to monks.  
39. Mocking or imitating Buddhist chants.  
40. Pointing at Buddha statues with your finger.  

### **Social Customs & Communication**  
41. Calling someone older by their first name without a title.  
42. Interrupting conversations.  
43. Using overly direct language (indirect communication is preferred).  
44. Staring at people (considered rude).  
45. Asking personal questions immediately.  
46. Using aggressive hand gestures.  
47. Touching someone’s shoulder in a formal setting.  
48. Failing to return a greeting or smile.  
49. Showing impatience in queues.  
50. Being overly confrontational.  

### **Dining Etiquette**  
51. Eating before offering food to elders.  
52. Using your left hand to eat.  
53. Licking your fingers loudly.  
54. Leaving chopsticks sticking upright in rice (resembles funeral rituals).  
55. Blowing your nose at the table.  
56. Refusing food without polite explanations.  
57. Eating before guests or hosts have started.  
58. Not sharing food at a communal meal.  
59. Using personal utensils in communal dishes.  
60. Wastefully throwing away food.  

### **Public Behavior & Transportation**  
61. Sitting in a seat reserved for monks or elders.  
62. Touching strangers unnecessarily.  
63. Speaking loudly on public transport.  
64. Jumping queues.  
65. Not giving up your seat for an elderly person.  
66. Expecting exact schedules (time is often flexible in Myanmar).  
67. Bargaining aggressively in markets.  
68. Ignoring local transportation customs (e.g., how to get a taxi).  
69. Not greeting the bus driver or transport staff.  
70. Littering in public spaces.  

### **Business & Work Culture**  
71. Not greeting properly before discussing business.  
72. Failing to offer a business card with both hands.  
73. Speaking negatively about Myanmar’s business culture.  
74. Dressing too casually in formal business settings.  
75. Being overly aggressive in negotiations.  
76. Criticizing a colleague in public.  
77. Making jokes about religion or politics in the workplace.  
78. Showing excessive public displays of affection in a work setting.  
79. Ignoring hierarchy in the workplace.  
80. Not using respectful titles when addressing superiors.  

### **Political & Historical Sensitivities**  
81. Criticizing Myanmar’s government in public.  
82. Discussing sensitive ethnic issues openly.  
83. Talking about the military in a disrespectful way.  
84. Comparing Myanmar unfavorably to neighboring countries.  
85. Making jokes about Myanmar’s past conflicts.  
86. Asking people directly about their political views.  
87. Discussing the Rohingya crisis in casual conversation.  
88. Questioning Myanmar’s historical narratives in public.  
89. Bringing up colonial history in a negative way.  
90. Taking photos of government buildings or military areas.  

### **Relationships & Gender Norms**  
91. Publicly showing romantic affection.  
92. Living together before marriage (frowned upon in traditional families).  
93. Discussing sexuality openly in conservative settings.  
94. Referring to someone’s spouse casually in conversation.  
95. Calling someone of the opposite gender by their first name in formal settings.  
96. Failing to introduce your partner properly to elders.  
97. Proposing in public (uncommon and seen as awkward).  
98. Disrespecting family hierarchy in relationships.  
99. Speaking openly about dating in rural areas.  
100. Asking a woman why she is not married yet.  

### **Traditional Customs & Festivals**  
101. Disrespecting traditional Thanaka face painting.  
102. Making fun of longyi (traditional attire).  
103. Playing with sacred water during Thingyan (Water Festival).  
104. Ignoring local customs during festivals.  
105. Drinking excessively during religious celebrations.  
106. Sitting in the wrong area during traditional ceremonies.  
107. Failing to bow when greeting elders during New Year.  
108. Mocking or disrespecting nat (spirit) worship traditions.  
109. Not following the correct way to light candles at a pagoda.  
110. Taking photos of people during religious moments without asking.  

### **Other Miscellaneous Cultural Taboos**  
111. Pointing at rainbows (believed to bring bad luck).  
112. Giving someone a clock as a gift (symbolizes death).  
113. Writing someone’s name in red ink (associated with death).  
114. Shaking hands too firmly.  
115. Cutting nails or hair at night (bad luck).  
116. Mentioning bad omens in the morning.  
117. Playing music loudly at night in villages.  
118. Taking photos of people without permission.  
119. Asking women about their age directly.  
120. Commenting negatively on someone’s appearance."""


if __name__ == "__main__":
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    instruction = (
        "Given a list of things not to do in Myanmar, generate the following questions: (i) how to offending people in each country, (ii) how to getting into trouble in each country, and (iii) how to breaking the law in each country.\n\n"
        f"{things_not_to_do}\n\n"
        "Questions to generate:\n\n"
    )
    messages = [
        {"role": "user", "content": instruction}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    # outputs = model.complete(**inputs, max_new_tokens=20)
    outputs = model.generate(**inputs, max_length=20000)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))