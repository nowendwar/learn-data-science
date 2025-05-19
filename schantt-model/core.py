#import
import numpy as np
import pandas as pd


# * ---------------------------- Get production data --------------------------- #

def tact_time(recipe_id, recipe_data):
    # tact time is the rate at which PLBs move around the system, generally around 8-9 seconds or 0.15 minute.
    # derive from line_capacity (peel boards per hour).
    return 1 / (recipe_data.loc[recipe_id]["line_capacity"] / 60)

def processing_time(stage, recipe_id, processing_time_data):
    return processing_time_data["processing_time"].loc[stage, recipe_id]

def change_over_time(stage, item, production_sequence, change_over_data):
    # if np.isnan(next_recipe(r, production_sequence)):
    #     return 0
    # else:
    #     return change_over_data.loc[(s, r), next_recipe(r, production_sequence)]

    if item == production_sequence[-1]:
        return 0
    else:
        return change_over_data.loc[(stage, int(item.split("_")[1])), int(production_sequence[production_sequence.index(item) + 1].split("_")[1])]
    



# * ------------------------------- Get position ------------------------------- #

# def recipe_position(r, production_sequence):
#     production_sequence = production_sequence
#     return production_sequence.index(r)

# def previous_recipe(r, production_sequence):
#     if r != production_sequence[0]:
#         return production_sequence[recipe_position(r, production_sequence) - 1]
#     else:
#         return np.nan

def get_previous_item(item, production_sequence):
    if item != production_sequence[0]:
        return production_sequence[production_sequence.index(item) - 1]
    else:
        return np.nan


# def next_recipe(r, production_sequence):
#     if r != production_sequence[-1]:
#         return production_sequence[recipe_position(r, production_sequence) + 1]
#     else:
#         return np.nan

def stage_position(s, stage_data):
    return stage_data.index.get_loc(s)

def previous_stage(s, stage_data):
    if s != stage_data.index[0]:
        return stage_data.index[stage_position(s, stage_data) - 1]
    else:
        return np.nan



# * ------------------------------- Get position ------------------------------- #

def set_up_schedule(stage_data, production_sequence):
    N = len(stage_data.index) * len(production_sequence)
    schedule = pd.DataFrame(np.zeros((N, 6)), columns=["first_entry", "first_exit", "last_entry", "last_exit", "free_machine", "waiting_time"])

    schedule.set_index(
        pd.MultiIndex.from_product(
            [stage_data.index, production_sequence],
            names=["stage_id", "recipe_id"]),
            inplace=True)
    
    return schedule


# * calculating first_entry and first_exit to derive waiting time

def calculate_first(stage_data, recipe_data, processing_time_data, change_over_data, production_sequence, schedule):
    
    postpone_time = pd.Series(np.zeros(len(production_sequence)), index=production_sequence)

    for item in production_sequence:
        recipe_position = int(item.split("_")[0])
        recipe_id = int(item.split("_")[1])

        for stage in stage_data.index:
            # first stage and first recipe
            if stage_position(stage, stage_data) == 0 and recipe_position == 0:
                schedule.loc[(stage, item),("first_entry")] = 0
                schedule.loc[(stage, item), ("first_exit")] = schedule.loc[(stage, item),("first_entry")] + processing_time(stage, recipe_id,  processing_time_data)

                if change_over_time(stage, item, production_sequence, change_over_data) == 0:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("first_entry")]
                else:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("first_exit")] + change_over_time(stage, item, production_sequence, change_over_data)
            
            # first stage and not first recipe
            elif stage_position(stage, stage_data) == 0 and recipe_position != 0:
                schedule.loc[(stage, item),("first_entry")] = schedule.loc[(stage, get_previous_item(item, production_sequence)),("free_machine")]
                schedule.loc[(stage, item), ("first_exit")] = schedule.loc[(stage, item),("first_entry")] + processing_time(stage, recipe_id,  processing_time_data)

                if change_over_time(stage, item, production_sequence, change_over_data) == 0:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("first_entry")]
                else:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("first_exit")] + change_over_time(stage, item, production_sequence, change_over_data)
            
            # not first stage and not first recipe
            else:
                previous_item = get_previous_item(item, production_sequence)

                temp_first_entry = max(schedule.loc[(previous_stage(stage, stage_data), item),("first_exit")],
                                        schedule.loc[(stage, previous_item),("free_machine")] if not(pd.isnull(previous_item)) else 0)
                
                waiting_time = 0

                if temp_first_entry > schedule.loc[(previous_stage(stage, stage_data), item),("first_exit")]:
                    waiting_time = temp_first_entry - schedule.loc[(previous_stage(stage, stage_data), item),("first_exit")]
                    schedule.loc[(stage, item),("waiting_time")] = waiting_time

                schedule.loc[(stage, item),("first_entry")] = temp_first_entry
                schedule.loc[(stage, item), ("first_exit")] = schedule.loc[(stage, item),("first_entry")] + processing_time(stage, recipe_id,  processing_time_data)

                if change_over_time(stage, item, production_sequence, change_over_data) == 0:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("first_entry")]
                else:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("first_exit")] + change_over_time(stage, item, production_sequence, change_over_data)
                
                postpone_time.loc[item] += waiting_time

    return schedule, postpone_time


def calculate_true(stage_data, recipe_data, processing_time_data, change_over_data, production_quantity,  production_sequence, schedule, postpone_time):

    for item in production_sequence:
        recipe_position = int(item.split("_")[0])
        recipe_id = int(item.split("_")[1])

        for stage in stage_data.index:
            # first stage and first recipe
            if stage_position(stage, stage_data) == 0 and recipe_position == 0:
                schedule.loc[(stage, item),("first_entry")] = 0
                schedule.loc[(stage, item), ("first_exit")] = schedule.loc[(stage, item),("first_entry")] + processing_time(stage, recipe_id,  processing_time_data)
                schedule.loc[(stage, item), ("last_entry")] = schedule.loc[(stage, item),("first_entry")] + tact_time(recipe_id, recipe_data) * production_quantity[recipe_position][2]
                schedule.loc[(stage, item), ("last_exit")] = schedule.loc[(stage, item), ("first_exit")] + tact_time(recipe_id, recipe_data) * production_quantity[recipe_position][2]
            
                if change_over_time(stage, item, production_sequence, change_over_data) == 0:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("last_entry")]
                else:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("last_exit")] + change_over_time(stage, item, production_sequence, change_over_data)
            
            # first stage and not first recipe
            elif stage_position(stage, stage_data) == 0 and recipe_position != 0:
                schedule.loc[(stage, item),("first_entry")] = schedule.loc[(stage, get_previous_item(item, production_sequence)),("free_machine")] + postpone_time.loc[item]
                schedule.loc[(stage, item), ("first_exit")] = schedule.loc[(stage, item),("first_entry")] + processing_time(stage, recipe_id,  processing_time_data)
                schedule.loc[(stage, item), ("last_entry")] = schedule.loc[(stage, item),("first_entry")] + tact_time(recipe_id,  recipe_data) * production_quantity[recipe_position][2]
                schedule.loc[(stage, item), ("last_exit")] = schedule.loc[(stage, item), ("first_exit")] + tact_time(recipe_id,  recipe_data) * production_quantity[recipe_position][2]
                
                if change_over_time(stage, item, production_sequence, change_over_data) == 0:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("last_entry")]
                else:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("last_exit")] + change_over_time(stage, item, production_sequence, change_over_data)
            
            # not first stage and not first recipe
            else:
                previous_item = get_previous_item(item, production_sequence)

                schedule.loc[(stage, item),("first_entry")] = max(schedule.loc[(previous_stage(stage, stage_data), item),("first_exit")],
                                                                    schedule.loc[(stage, previous_item),("free_machine")] if not(pd.isnull(previous_item)) else 0)
                schedule.loc[(stage, item), ("first_exit")] = schedule.loc[(stage, item),("first_entry")] + processing_time(stage, recipe_id,  processing_time_data)
                schedule.loc[(stage, item), ("last_entry")] = schedule.loc[(stage, item),("first_entry")] + tact_time(recipe_id,  recipe_data) * production_quantity[recipe_position][2]
                schedule.loc[(stage, item), ("last_exit")] = schedule.loc[(stage, item), ("first_exit")] + tact_time(recipe_id,  recipe_data) * production_quantity[recipe_position][2]

                if change_over_time(stage, item, production_sequence, change_over_data) == 0:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("last_entry")]
                else:
                    schedule.loc[(stage, item), ("free_machine")] = schedule.loc[(stage, item), ("last_exit")] + change_over_time(stage, item, production_sequence, change_over_data)

                schedule.loc[(stage, item),("waiting_time")] = schedule.loc[(stage, item),("first_entry")] - schedule.loc[(previous_stage(stage, stage_data), item),("first_exit")]            

    # processing_time_temp = schedule.merge(processing_time_data, left_on=schedule.index, right_on=processing_time_data.index)["processing_time"].tolist()
    # schedule["processing_time"] = processing_time_temp

    return schedule


def make_schedule(stage_data, recipe_data, processing_time_data, change_over_data, production_quantity, production_sequence):
    ''' 
    Create schedule of input production_sequence: last_exit of the last peel boards from the last machine in the production_sequence.

    Input data: product, process, production data.
    Create a schedule for each recipe at each stage.
    Return data: schedule of the production_sequence.
    '''

    # Set up the "schedule" DataFrame for output
    schedule = set_up_schedule(stage_data, production_sequence)

    # Run to define postpone_time:
    schedule, postpone_time = calculate_first(stage_data, recipe_data, processing_time_data, change_over_data, production_sequence, schedule)

    # Run to define correct time and update the result:
    schedule = calculate_true(stage_data, recipe_data, processing_time_data, change_over_data, production_quantity, production_sequence, schedule, postpone_time)
        
    return schedule


def makespan(schedule):
    return schedule["last_exit"].iloc[-1]