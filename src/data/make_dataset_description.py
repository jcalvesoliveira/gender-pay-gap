# -*- coding: utf-8 -*-
import logging
from posixpath import split

import click
import pandas as pd

TEMPLATE = """
1. Title: {0}

2. Sources:
    ----------------EXAMPLE--------------------------------------
   (a) Creator: Marko Bohanec
   (b) Donors: Marko Bohanec   (marko.bohanec@ijs.si)
               Blaz Zupan      (blaz.zupan@ijs.si)
   (c) Date: June, 1997
    ----------------EXAMPLE--------------------------------------

3. Past Usage:

    ----------------EXAMPLE--------------------------------------
   The hierarchical decision model, from which this dataset is
   derived, was first presented in 

   M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for
   multi-attribute decision making. In 8th Intl Workshop on Expert
   Systems and their Applications, Avignon, France. pages 59-78, 1988.

   Within machine-learning, this dataset was used for the evaluation
   of HINT (Hierarchy INduction Tool), which was proved to be able to
   completely reconstruct the original hierarchical model. This,
   together with a comparison with C4.5, is presented in

   B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by
   function decomposition. ICML-97, Nashville, TN. 1997 (to appear)
   ----------------EXAMPLE--------------------------------------

4. Relevant Information Paragraph:

   ----------------EXAMPLE--------------------------------------
   Car Evaluation Database was derived from a simple hierarchical
   decision model originally developed for the demonstration of DEX
   (M. Bohanec, V. Rajkovic: Expert system for decision
   making. Sistemica 1(1), pp. 145-157, 1990.). The model evaluates
   cars according to the following concept structure:

   CAR                      car acceptability
   . PRICE                  overall price
   . . buying               buying price
   . . maint                price of the maintenance
   . TECH                   technical characteristics
   . . COMFORT              comfort
   . . . doors              number of doors
   . . . persons            capacity in terms of persons to carry
   . . . lug_boot           the size of luggage boot
   . . safety               estimated safety of the car

   Input attributes are printed in lowercase. Besides the target
   concept (CAR), the model includes three intermediate concepts:
   PRICE, TECH, COMFORT. Every concept is in the original model
   related to its lower level descendants by a set of examples (for
   these examples sets see http://www-ai.ijs.si/BlazZupan/car.html).

   The Car Evaluation Database contains examples with the structural
   information removed, i.e., directly relates CAR to the six input
   attributes: buying, maint, doors, persons, lug_boot, safety.

   Because of known underlying concept structure, this database may be
   particularly useful for testing constructive induction and
   structure discovery methods.
   ----------------EXAMPLE--------------------------------------

5. Number of Instances: {1}
   (instances completely cover the attribute space)

6. Number of Attributes: {2}

7. Attribute Values:

{3}

8. Missing Attribute Values: {4}

9. Class Distribution (number of instances per class)

{5}
"""


class DatasetDescription:

    def __init__(self, input_filepath: str):
        self.input_filepath = input_filepath
        self.df = pd.read_csv(input_filepath)

    def get_name(self) -> str:
        filename = self.input_filepath.split("/")[-1].split(".")[0]
        return filename

    def get_instances_count(self) -> int:
        return self.df.shape[0]

    def get_attributes_count(self) -> int:
        return self.df.shape[1]

    def get_missing_atributes(self) -> int:
        return self.df.isnull().sum().sum()

    def get_attribute_values(self) -> str:
        result = ""
        for column in self.df.columns:
            unique_values = [
                x if type(x) == str else str(x)
                for x in self.df[column].unique()
            ]
            if len(unique_values) > 10:
                result += f"   {column}: {', '.join(unique_values[0:9])}, ...\n"
            else:
                result += f"   {column}: {', '.join(unique_values)}\n"
        return result

    def get_class_distribution(self) -> str:
        result = ""
        for column in self.df.columns:
            data_type = self.df[column].dtype
            if data_type in ("object", "bool"):
                result += "\n------------------------------------------------------\n"
                result += f"\n   class      {column}          {column}[%]"
                for class_ in self.df[column].unique():
                    class_counts = self.df[self.df[column] == class_].shape[0]
                    total_counts = self.df.shape[0]
                    result += f"   {class_} {class_counts} ({class_counts/total_counts*100:.2f} %)\n"

        return result

    def to_file(self, output_filepath: str):
        file_name = self.input_filepath.split("/")[-1].split(".")[0]
        with open(f"{output_filepath}{file_name}.names", "w") as f:
            f.write(str(self))

    def __str__(self):
        return TEMPLATE.format(self.get_name(), self.get_instances_count(),
                               self.get_attributes_count(),
                               self.get_attribute_values(),
                               self.get_missing_atributes(),
                               self.get_class_distribution())


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    DatasetDescription(input_filepath).to_file(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main('data/external/CurrentPopulationSurvey.csv', 'docs/datasets/')
