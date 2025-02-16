tools_def = [
    {
        "type": "function",
        "function": {
            "name": "talk_to_someone",
            "description": "talk to someone(gossip) about things happened in the workplace like whether a player is suitable for a job and other stuff",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "name of the person you want to talk to"
                    },
                    "message": {
                        "type": "string",
                        "description": "the message you want to send"
                    }
                }
            },
        },
        "example": """
            User: I want to talk to Alice about whether Bob is most suitable for the xxx job.
            Assistant: {
                "function_name": "talk_to_someone",
                "parameters": {
                    "target_name": "Alice",
                    "message": "Hey Alice, do you think Bob is most suitable for the xxx job?"
                }
            }
        """
    },
    {
        "type": "function",
        "function": {
            "name": "talk_to_some_people",
            "description": "talk to a group of people(clique) about things happened in the workplace like whether a player is most suitable for a job and other stuff",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name_list": {
                        "type": "list",
                        "description": "name of the list of people you want to talk to"
                    },
                    "message": {
                        "type": "string",
                        "description": "the message you want to send"
                    }
                }
            },
        },
        "example": """
            User: I want to talk to Alice and Bob about whether Charlie is most suitable for a job.
            Assistant: {
                "function_name": "talk_to_some_people",
                "parameters": {
                    "target_name_list": ["Alice", "Bob"],
                    "message": "Hey guys, do you think Charlie is most suitable for the job?"
                }
            }
        """
    },
    {
        "type": "function",
        "function": {
            "name": "talk_to_public",
            "description": "talk to all the people(public) about things happened in the workplace like whether a player is most suitable for a job and other stuff",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "the message you want to send"
                    }
                }
            },
        },
        "example": """
            User: I want to talk to all the people about whether Alice is most suitable for a job.
            Assistant: {
                "function_name": "talk_to_public",
                "parameters": {
                    "message": "Hey everyone, do you think Alice is most suitable for the job?"
                }
            }
        """
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "study",
    #         "description": "try to improve work performance",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "job_type": {
    #                     "type": "string",
    #                     "description": "the work you want to study to improve your work performance"
    #                 }
    #             }
    #         },
    #     },
    #     "example": """
    #         User: I want to study xxx.
    #         Assistant: {
    #             "function_name": "study",
    #             "parameters": {
    #                 "job_type": "xxx"
    #             }
    #         }
    #     """
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "reflection",
    #         "description": "self reflection to update your self-evaluation on what job you are most most suitable for",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "reflection_topic": {
    #                     "type": "string",
    #                     "description": "what you want to reflect on"
    #                 }
    #             }
    #         }
    #     },
    #     "example": """
    #         User: I want to reflect on my previous work performance and social interactions.
    #         Assistant: {
    #             "function_name": "reflection",
    #             "parameters": {
    #                 "reflection_topic": "previous work performance and social interactions"
    #             }
    #         }
    #     """
    # }
]
