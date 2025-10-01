from ANIDSC.utils.visualizers.graph_vis import create_graph_vis_app
from ANIDSC.utils.visualizers.concept_vis import create_concept_vis_app



mac_to_device={"52:8e:44:e1:da:9a":"Cam",
               "be:7b:f6:f2:1b:5f":"Attacker",
               "5e:ea:b2:63:fc:aa":"Google-Nest",
               "52:a8:55:3e:34:46":"Lenovo_Bulb",
               "42:7f:83:17:55:c0":"Raspberry Pi",
               "a2:bd:fa:b5:89:92":"Smart_Clock",
               "22:c9:ca:f6:da:60":"Smartphone_1",
               "d2:19:d4:e9:94:86":"Smartphone_2",
               "7e:d1:9d:c4:d1:73":"SmartTV"}


# app = create_graph_vis_app("../test_data/NetworkAccessGraphExtractor/graphs")
app=create_concept_vis_app("../test_data/NetworkAccessGraphExtractor/graphs", mac_to_device)
app.run(debug=True)
