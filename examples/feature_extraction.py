from ANIDSC.templates import extract_network_features

if __name__ == "__main__":
    # targetted attacks
    # devices=["Smart_TV","Cam_1","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1","Smartphone_1","Smartphone_2","Lenovo_Bulb_1",]
    # attacks=["ACK_Flooding", "SYN_Flooding", "UDP_Flooding", "Port_Scanning", "Service_Detection"]

    cic_files = [
        "Monday-WorkingHours",
        "Tuesday-WorkingHours",
        "Wednesday-WorkingHours",
        "Thursday-WorkingHours",
        "Friday-WorkingHours",
    ]
    for i, file in enumerate(cic_files):
        if i > 0:
            state = cic_files[i - 1]
        else:
            state = None
            
        extract_network_features(
            "CIC_IDS_2017",
            "AfterImageGraph",
            {
                "graph_type": "multi_layer",
                "protocols": ["TCP", "DNS", "SSH", "FTP", "HTTP","UDP","HTTPS"],
            },
            file,
            state=state,
        )

        # find_concept_drift_times(
        #     "CIC_IDS_2017",
        #     "AfterImageGraph",
        #     f"{file}_meta",
        #     "Canada/Atlantic",
        #     {
        #         1: [
        #             [time(9, 20, 0), time(10, 20, 0)],
        #             [time(14, 0, 0), time(15, 0, 0)],
        #         ],
        #         2: [
        #             [time(9, 47, 0), time(10, 10, 0)],
        #             [time(10, 14, 0), time(10, 35, 0)],
        #             [time(10, 43, 0), time(11, 0, 0)],
        #             [time(11, 10, 0), time(11, 23, 0)],
        #             [time(15, 12, 0), time(15, 32, 0)],
        #         ],
        #         3: [
        #             [time(9, 20, 0), time(10, 0, 0)],
        #             [time(10, 15, 0), time(10, 35, 0)],
        #             [time(10, 40, 0), time(10, 42, 0)],
        #             [time(14, 19, 0), time(14, 35, 0)],
        #             [time(14, 53, 0), time(15, 0, 0)],
        #             [time(15, 4, 0), time(15, 45, 0)],
        #         ],
        #         4: [
        #             [time(10, 2, 0), time(11, 2, 0)],
        #             [time(13, 55, 0), time(14, 35, 0)],
        #             [time(14, 51, 0), time(15, 29, 0)],
        #             [time(15, 56, 0), time(16, 16, 0)],
        #         ],
        #     },
        # )
        # print("-" * 50)
    #

    # synthetic_features("FakeGraphData","SyntheticFeatureExtractor")
