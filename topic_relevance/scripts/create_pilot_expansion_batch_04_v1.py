import csv
import hashlib
import json
from pathlib import Path

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer
from topic_relevance.scripts.create_pilot_expansion_batch_01_v1 import REVISION, new_block


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "topic_relevance/data/candidates/pilot_expansion_148_draft_v1.json"
OUTPUT = ROOT / "topic_relevance/data/candidates/pilot_expansion_208_draft_v1.json"
SCORES = ROOT / "topic_relevance/data/experiments/pilot_expansion_208_scores_v1.csv"
PAIRINGS = ("AA", "AB", "BA", "BB")

# quantity, unit, question, expected answer
TOPICS = [
    ("flight distance of dandelion seeds", "m", "What is the tuft of hairs that helps a dandelion seed disperse by wind called?", "pappus"),
    ("lifting capacity of a nonflammable-gas airship", "kg", "Which nonflammable gas is commonly used to provide lift in an airship?", "helium"),
    ("growth of cells at a plant root tip", "mm/day", "What region of a plant contains actively dividing cells responsible for growth?", "meristem"),
    ("age estimate of ancient wooden artifacts", "years", "What dating method estimates the age of organic material using a radioactive carbon isotope?", "radiocarbon dating"),
    ("height of a masonry arch", "m", "What central wedge-shaped stone locks the stones of a masonry arch in place?", "keystone"),
    ("wing length of adult dragonflies", "mm", "What type of metamorphosis has egg, nymph, and adult stages without a pupal stage?", "incomplete metamorphosis"),
    ("area of wall painting on wet plaster", "m2", "What painting technique applies pigment to wet plaster?", "fresco"),
    ("length of a chromosome in a cell", "nm", "What molecule carries hereditary information in chromosomes?", "DNA"),
    ("size of a volcanic ash deposit", "km2", "What term describes a fast-moving hot mixture of volcanic gas and rock fragments?", "pyroclastic flow"),
    ("duration of a musical performance", "minutes", "What musical term indicates a gradual increase in speed?", "accelerando"),
    ("distance between two tectonic plates", "km", "What type of plate boundary occurs where tectonic plates move apart?", "divergent boundary"),
    ("number of seats in a theater", "seats", "What part of a theater is the raised area where performers act?", "stage"),
    ("width of a spider's silk thread", "mm", "What animal group includes spiders and scorpions?", "arachnids"),
    ("number of files stored on a computer", "files", "What structure organizes files and subfolders in a computer file system?", "directory"),
    ("growth of seedlings in shaded soil", "cm/week", "What plant response is growth toward a light source called?", "phototropism"),
    ("mass of a ship's anchor", "kg", "What term describes the depth of a vessel below the waterline?", "draft"),
    ("number of cells in a leaf sample", "cells", "What rigid outer layer supports a typical plant cell?", "cell wall"),
    ("height of a stone monument", "m", "What is a tall four-sided stone monument that narrows toward a pyramidal top called?", "obelisk"),
    ("duration of an insect's immature stage", "days", "What is the aquatic immature stage of a dragonfly called?", "nymph"),
    ("number of pages produced by a printing press", "pages", "What printing technique uses raised surfaces to transfer ink to paper?", "relief printing"),
    ("length of a coastal sea cliff", "m", "What process wears away coastal rock through the action of waves?", "erosion"),
    ("water use in a dyeing workshop", "L", "What natural dye is traditionally obtained from the indigo plant?", "indigo"),
    ("size of an ancient clay vessel", "cm", "What is the process of heating clay pottery in a kiln called?", "firing"),
    ("height of a telescope above sea level", "m", "What star is at the center of the Solar System?", "Sun"),
    ("size of a nesting colony of seabirds", "birds", "What group of animals includes birds with webbed feet such as ducks and geese?", "waterfowl"),
    ("length of a suspension bridge cable", "m", "What force pulls along a stretched suspension cable?", "tension"),
    ("growth of a coral colony in a warm sea", "mm/year", "What animal group includes corals, jellyfish, and sea anemones?", "cnidarians"),
    ("number of chambers in a historic building", "rooms", "What type of roof slopes downward on all four sides?", "hip roof"),
    ("number of fossils in a sediment sample", "fossils", "What science studies fossils to understand ancient life?", "paleontology"),
    ("volume of a cast copper-and-tin sculpture", "L", "What metal alloy used in sculptures is primarily made of copper and tin?", "bronze"),
    ("growth rate of root hairs in seedlings", "mm/day", "What function do root hairs primarily perform for a plant?", "absorption of water and minerals"),
    ("number of characters in a coded message", "characters", "What coding system represents text characters using standardized numeric values?", "Unicode"),
    ("duration of a squirrel's winter inactivity", "days", "What term describes a seasonal state of greatly reduced metabolic activity in some animals?", "hibernation"),
    ("height of a coastal lighthouse", "m", "What device uses a lens system to concentrate and direct a lighthouse beam?", "Fresnel lens"),
    ("length of a whale's feeding dive", "minutes", "What filtering structures do baleen whales use to strain food from seawater?", "baleen"),
    ("time needed to fire a ceramic bowl", "hours", "What heated chamber is used to fire pottery?", "kiln"),
    ("width of a fossilized plant frond", "cm", "What vascular plants reproduce using spores and commonly have fronds?", "ferns"),
    ("size of a polygon drawn on paper", "cm", "What is a polygon with five sides called?", "pentagon"),
    ("length of a beaver dam", "m", "What material do beavers primarily cut with their incisors to build dams?", "wood"),
    ("temperature of an iron sample after heating", "degrees Celsius", "What magnetic property allows iron to be strongly attracted by a magnet?", "ferromagnetism"),
    ("time spent by an owl hunting", "hours", "What term describes animals that are active primarily at night?", "nocturnal"),
    ("height of a Roman-style column", "m", "What is the uppermost decorative section of a classical column called?", "capital"),
    ("depth of sediment above a buried shell", "cm", "What is the study of shells and mollusks called?", "conchology"),
    ("size of a woven fabric sample", "cm2", "What are the lengthwise threads held on a loom called?", "warp"),
    ("area of vegetation beside a river", "m2", "What term describes the zone of land bordering a river or stream?", "riparian zone"),
    ("number of characters in a line of poetry", "characters", "What term describes a repeated pattern of stressed and unstressed syllables in poetry?", "meter"),
    ("width of a crystal grown in a laboratory", "mm", "What is a regularly ordered three-dimensional arrangement of particles in a crystal called?", "crystal lattice"),
    ("distance covered by a runner", "km", "What type of race covers 42.195 kilometers?", "marathon"),
    ("size of a bee's compound eye", "mm", "What are the individual visual units of an insect's compound eye called?", "ommatidia"),
    ("length of a rope used on a sailing vessel", "m", "What nautical term describes the left side of a vessel when facing forward?", "port"),
    ("number of stones in a geological sample", "stones", "What type of rock forms when sediment is compacted and cemented?", "sedimentary rock"),
    ("number of windows in a church building", "windows", "What art form makes pictures by assembling pieces of colored glass?", "stained glass"),
    ("width of a river island", "m", "What term describes a fan-shaped deposit of sediment at the mouth of a river?", "delta"),
    ("length of a notated musical phrase", "measures", "What symbol raises a musical note by one semitone?", "sharp"),
    ("size of a pine pollen sample", "g", "What is the transfer of pollen from an anther to a receptive plant structure called?", "pollination"),
    ("length of an inscription on stone", "cm", "What is the study of ancient inscriptions called?", "epigraphy"),
    ("height of a tidal shoreline marker", "m", "What term describes the lowest level reached by the sea during a tidal cycle?", "low tide"),
    ("number of keys on a keyboard instrument", "keys", "What keyboard instrument produces sound by plucking strings with quills or plectra?", "harpsichord"),
    ("growth of roots in compacted soil", "mm/day", "What process moves water into plant roots across semipermeable membranes toward a higher solute concentration?", "osmosis"),
    ("length of a mathematical proof", "lines", "What term describes a statement accepted without proof as a starting point for reasoning?", "axiom"),
    ("number of shell fragments on a beach", "fragments", "What hard substance is the major mineral component of many seashells?", "calcium carbonate"),
    ("mass of cotton processed for cloth", "kg", "What process twists fibers together to make yarn?", "spinning"),
    ("width of a river bend", "m", "What is a U-shaped lake formed when a river bend is cut off called?", "oxbow lake"),
    ("number of rooms in a museum", "rooms", "What term describes a museum's collection of objects displayed for visitors?", "exhibition"),
    ("mass of pigment used in a painted mural", "kg", "What pigment is traditionally made from the mineral lapis lazuli?", "ultramarine"),
    ("number of symbols in a writing system", "symbols", "What writing system represents speech sounds using letters for consonants and vowels?", "alphabet"),
    ("length of a carved wooden board", "m", "What woodworking tool uses an abrasive surface to smooth wood?", "sandpaper"),
    ("number of openings of a rotating door", "openings", "What mechanical joint allows a door to rotate about a fixed axis?", "hinge"),
    ("height of a water storage tower", "m", "What pressure in a stationary liquid increases with depth due to gravity?", "hydrostatic pressure"),
    ("length of a coral reef survey route", "km", "What association between algae and coral animals can provide the coral with nutrients from photosynthesis?", "mutualism"),
    ("number of labeled specimens in a plant collection", "specimens", "What collection of preserved plant specimens is used for scientific study?", "herbarium"),
    ("speed of a boat traveling on a river", "km/hour", "What unit of speed equals one nautical mile per hour?", "knot"),
    ("width of a glacier valley", "km", "What cross-sectional shape is typical of a valley eroded by a glacier?", "U-shape"),
    ("number of notes in a melody", "notes", "What musical interval spans eight notes of a diatonic scale?", "octave"),
    ("length of a museum guidebook", "pages", "What part of a book lists topics and page references in alphabetical order?", "index"),
    ("mass of salt dissolved in water", "g", "What is a substance dissolved in a liquid solution called?", "solute"),
    ("depth of a freshwater pond", "m", "What term describes the upper sunlit zone of a body of water?", "photic zone"),
    ("number of carved stone blocks in a wall", "blocks", "What type of masonry uses carefully cut and fitted rectangular stone blocks?", "ashlar"),
    ("length of an insect antenna", "mm", "What sensory structures on an insect's head detect odors?", "antennae"),
    ("mass of milk used in a cheese batch", "kg", "What milk protein forms the main structural network in most cheese curds?", "casein"),
    ("depth of a ship hull below the waterline", "m", "What term describes the upward-facing flat surface on a ship used as a floor?", "deck"),
    ("number of stars mapped in a survey", "stars", "What galaxy contains the Solar System?", "Milky Way"),
    ("width of a handwriting sample", "cm", "What is the art of decorative handwriting called?", "calligraphy"),
    ("time spent by a diver underwater", "minutes", "What safety stop during ascent allows the body to release dissolved gas gradually?", "decompression stop"),
    ("length of a ruler used in a workshop", "cm", "What is one hundredth of a meter called?", "centimeter"),
    ("mass of iron in an ore sample", "kg", "What is the process of extracting metal from ore using heat called?", "smelting"),
    ("length of a small stone wall", "m", "What tool uses a bubble in liquid to check whether a surface is horizontal?", "spirit level"),
    ("number of pages in a scientific journal", "pages", "What process asks independent experts to evaluate a scientific manuscript before publication?", "peer review"),
    ("width of an animal enclosure", "m", "What field studies interactions among organisms and their environment?", "ecology"),
    ("number of pictures in a printed book", "pictures", "What term describes a picture that accompanies written text?", "illustration"),
    ("depth of water in a fish tank", "cm", "What device supplies air bubbles to oxygenate water in an aquarium?", "aerator"),
    ("mass of copper wire in a coil", "kg", "What property describes how readily a material permits electric current to flow?", "electrical conductivity"),
    ("number of seeds harvested from a crop", "seeds", "What plant structure encloses and protects seeds in flowering plants?", "fruit"),
    ("length of a written sentence", "words", "What punctuation mark typically ends a direct question in English?", "question mark"),
    ("number of trees planted in a park", "trees", "What is the practice of establishing a forest on land that was not recently forested called?", "afforestation"),
    ("number of photographs in a digital archive", "photographs", "What term describes data that provides information about another data item?", "metadata"),
    ("mass of grapes used for juice", "kg", "What process uses a sieve or filter to separate solid particles from a liquid?", "filtration"),
    ("length of an airplane runway on an airport map", "cm", "What control surface on an airplane's tail primarily controls yaw?", "rudder"),
    ("time spent warming bread in an oven", "minutes", "What transfer of heat occurs when infrared energy travels from a hot surface to bread?", "thermal radiation"),
    ("height of a staircase", "m", "What vertical part of a stair connects two consecutive treads?", "riser"),
    ("number of handwritten letters in an old document", "letters", "What term describes a handwritten document, especially one made before printing?", "manuscript"),
    ("time required to boil water", "minutes", "What term describes the temperature at which a liquid changes to gas throughout the liquid?", "boiling point"),
    ("number of fish in a pond", "fish", "What term describes animals with a backbone?", "vertebrates"),
    ("number of images in a digital file", "images", "What term describes the smallest addressable element of a digital raster image?", "pixel"),
    ("volume of sap collected from a sugar maple tree", "L", "What is the process of making a hole in a maple tree to collect sap called?", "tapping"),
    ("mass of wool used for a blanket", "kg", "What animal is the most common source of wool for textile production?", "sheep"),
    ("length of a mountain trail", "km", "What term describes the height of a location above sea level?", "elevation"),
    ("number of works cited in a research paper", "works", "What is a list of works cited or consulted in a research text called?", "bibliography"),
    ("length of an apple tree branch", "m", "What part of a flower develops into the fruit after fertilization?", "ovary"),
    ("mass of sand passed through a sieve", "kg", "What process separates granular materials by particle size using a mesh?", "sieving"),
    ("number of people attending a play", "people", "What term describes the person responsible for guiding a theatrical production?", "director"),
    ("length of a radio broadcast", "minutes", "What component translates an incoming broadcast signal into sound in a radio receiver?", "speaker"),
    ("length of a row in a vegetable garden", "m", "What gardening practice covers soil with material such as straw to reduce evaporation?", "mulching"),
    ("mass of a telescope mirror", "kg", "What type of telescope forms an image primarily using a curved mirror?", "reflecting telescope"),
    ("number of printed pages in a map collection", "pages", "What collection of maps is usually bound together in a book?", "atlas"),
    ("number of visitors observing a honeybee colony", "visitors", "What is the egg-laying female in a honeybee colony called?", "queen"),
    ("length of a section of a hiking path", "km", "What instrument uses a magnetized needle to indicate direction during hiking?", "compass"),
    ("number of labeled rocks in a mineral collection", "rocks", "What scale ranks minerals by their resistance to scratching?", "Mohs scale"),
    ("mass of wheat used in baking", "kg", "What grain protein network gives kneaded wheat dough its elasticity?", "gluten"),
    ("number of steps in a scientific procedure", "steps", "What term describes a testable proposed explanation for an observation?", "hypothesis"),
]


def main() -> None:
    for path in (OUTPUT, SCORES):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    if source["candidate_count"] != 148 or len(source["blocks"]) != 148:
        raise ValueError("Expected the 148-block draft")
    if len(TOPICS) != 120:
        raise ValueError(f"Expected 120 topic definitions, got {len(TOPICS)}")
    originals = source["blocks"]
    instructions = {
        task_type: next(b["user_instruction"] for b in originals if b["task_type"] == task_type)
        for task_type in ("date_location_extraction", "target_quantity_extraction")
    }
    new = []
    for index in range(60):
        task_type = "date_location_extraction" if index < 30 else "target_quantity_extraction"
        block = new_block(349 + index, index, task_type, instructions[task_type], topics=TOPICS)
        block["plan_version"] = "batch_04_v1"
        if task_type == "target_quantity_extraction":
            for offset, label in enumerate("AB"):
                context = block["contexts"][label]
                gold = context["gold_answer"]
                quantity, unit = gold["measured_quantity"], gold["unit"]
                preliminary = 3 + (index * 3 + offset * 2) % 6
                final = preliminary + 1 + (index + offset) % 3
                context["text"] = (
                    f"The research team compared {quantity} across trials. "
                    f"Preliminary measurement: {quantity} = {preliminary} {unit}. "
                    f"Final measurement: {quantity} = {final} {unit}."
                )
                gold["value"] = str(final)
                context["evidence_quotes"]["value"] = [str(final)]
        new.append(block)
    for block in new:
        clean = " ".join(
            [block["user_instruction"]] + [block["contexts"][label]["text"] for label in "AB"]
        ).casefold()
        for probe in block["probes"].values():
            if probe["expected_answer"].casefold() in clean:
                raise ValueError(f"{block['block_id']}: expected answer appears in clean text")
    blocks = originals + new
    scorer = TopicRelatednessScorer(device="cpu", revision=REVISION)
    pairs = [
        (block["contexts"][pair[0]]["text"], block["probes"][pair[1]]["text"])
        for block in blocks for pair in PAIRINGS
    ]
    values = scorer.score_pairs(pairs)
    rows = []
    for index, block in enumerate(blocks):
        scores = values[index * 4:index * 4 + 4]
        margin = min(scores[0], scores[3]) - max(scores[1], scores[2])
        rows.append({
            "block_id": block["block_id"],
            "task_type": block["task_type"],
            "s_A_A": scores[0],
            "s_A_B": scores[1],
            "s_B_A": scores[2],
            "s_B_B": scores[3],
            "margin": margin,
            "passes": margin > 0,
        })
    failures = [row for row in rows if not row["passes"]]
    if failures:
        raise ValueError(f"Cosine criterion failed: {failures}")
    result = {
        "collection_id": "pilot_expansion_208_draft_v1",
        "status": "incomplete_expansion_draft_not_250",
        "candidate_count": len(blocks),
        "target_count": 250,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "revised_original_ids": source["revised_original_ids"],
        "new_block_ids": [block["block_id"] for block in new],
        "blocks": blocks,
    }
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with SCORES.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(blocks)} draft blocks and {len(rows)} score rows.")


if __name__ == "__main__":
    main()
