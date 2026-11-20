import csv
import hashlib
import json
from pathlib import Path

from topic_relevance.scorer.topic_relatedness import TopicRelatednessScorer
from topic_relevance.scripts.create_pilot_expansion_batch_01_v1 import REVISION, new_block


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "topic_relevance/data/candidates/pilot_expansion_208_draft_v1.json"
OUTPUT = ROOT / "topic_relevance/data/candidates/pilot_expansion_250_draft_v1.json"
SCORES = ROOT / "topic_relevance/data/experiments/pilot_expansion_250_scores_v1.csv"
PAIRINGS = ("AA", "AB", "BA", "BB")

# quantity, unit, question, expected answer
TOPICS = [
    ("length of ancient Egyptian reed writing sheets", "cm", "What writing material did ancient Egyptians make from the pith of a marsh plant?", "papyrus"),
    ("number of planets found orbiting distant stars", "planets", "What is a planet orbiting a star outside the Solar System called?", "exoplanet"),
    ("number of piano strings struck during a performance", "strings", "What part of a piano strikes a string to produce a note?", "hammer"),
    ("area of farmland planted without soil tillage", "hectares", "What farming practice leaves soil largely undisturbed when planting crops?", "no-till farming"),
    ("number of words translated between languages", "words", "What is a word borrowed from another language called?", "loanword"),
    ("number of coins minted from precious metal", "coins", "What precious yellow metal has the chemical symbol Au?", "gold"),
    ("number of currencies traded in an international market", "currencies", "What is the rate at which one currency can be exchanged for another called?", "exchange rate"),
    ("number of stars in a recognized night-sky pattern", "stars", "What is a named pattern or region of stars in the night sky called?", "constellation"),
    ("length of an ancient stone road", "km", "What empire built an extensive network of paved roads across much of Europe and the Mediterranean?", "Roman Empire"),
    ("number of cranial bones in a mammal specimen", "bones", "What protective bony structure encloses the brain in a vertebrate?", "skull"),
    ("size of a collection of postal stamps", "stamps", "What is the collection and study of postage stamps called?", "philately"),
    ("width of a decorative ceramic tile picture", "cm", "What art form makes a picture by arranging small pieces of stone or colored tile?", "mosaic"),
    ("number of court cases heard by a judge", "cases", "What term describes a previous court decision used as a guide in later cases?", "precedent"),
    ("number of tree species found in a forest", "species", "What measure counts the number of different species in an ecological community?", "species richness"),
    ("duration of a chess tournament match", "minutes", "What chess move places the opposing king under immediate attack?", "check"),
    ("length of a written computer program", "lines", "What type of software converts source code into machine code?", "compiler"),
    ("number of printed pages in a dictionary", "pages", "What is the study of the origins and history of words called?", "etymology"),
    ("height of airborne debris from an explosive volcanic eruption", "m", "What is the cloud of fine rock particles ejected by a volcanic eruption called?", "volcanic ash cloud"),
    ("area of land occupied by grape vines", "hectares", "What agricultural practice is the cultivation of grape vines called?", "viticulture"),
    ("number of cycles of a weight-driven longcase clock", "cycles", "What swinging weight regulates the movement of a traditional longcase clock?", "pendulum"),
    ("number of symbols in an ancient script", "symbols", "What ancient Egyptian writing system used pictorial signs?", "hieroglyphs"),
    ("duration of a film shown in a cinema", "minutes", "What film technique joins separate shots together to create a sequence?", "editing"),
    ("height of a wind-shaped sand formation", "m", "What is a crescent-shaped sand dune with horns pointing downwind called?", "barchan"),
    ("number of words in a spoken pronunciation exercise", "words", "What sound unit contains a vowel sound and may also contain consonants?", "syllable"),
    ("number of ocean animals with radial symmetry", "animals", "What marine animal has five arms and belongs to the echinoderm group?", "sea star"),
    ("number of citations in a journal article", "citations", "What term describes the reuse of someone else's words or ideas without proper attribution?", "plagiarism"),
    ("number of customers in a market", "customers", "What economic term describes the amount of a product consumers are willing and able to buy?", "demand"),
    ("duration of an observation of Earth's shadow crossing the Moon", "minutes", "What kind of eclipse occurs when Earth passes between the Sun and the Moon?", "lunar eclipse"),
    ("length of a chromosome region in a genetics study", "nm", "What is a version of a gene called?", "allele"),
    ("number of columns in a table of numerical data", "columns", "What statistical value is found by adding observations and dividing by their count?", "arithmetic mean"),
    ("number of pottery fragments found at a dig site", "fragments", "What is the scientific study of the human past through material remains called?", "archaeology"),
    ("length of a printed musical score", "pages", "What is the symbol at the beginning of a musical staff that assigns pitches to lines called?", "clef"),
    ("number of mushrooms examined in a forest survey", "mushrooms", "What network of fungal filaments forms the main body of many fungi?", "mycelium"),
    ("number of travelers on a passenger train", "travelers", "What term describes the planned sequence of stops and travel times for a train?", "timetable"),
    ("area of ice cover at the North Pole", "km2", "What ocean surrounds the North Pole?", "Arctic Ocean"),
    ("length of a poem read aloud", "lines", "What is a group of lines in a poem called?", "stanza"),
    ("number of metal samples tested for magnetism", "samples", "What metal is liquid at typical room temperature?", "mercury"),
    ("length of an ancient water channel for crops", "km", "What agricultural technique supplies water to crops through artificial channels?", "irrigation"),
    ("number of colored flags displayed on a ship", "flags", "What international maritime signaling system uses colored flags to represent letters?", "International Code of Signals"),
    ("area of a painted cave wall", "m2", "What term describes the prehistoric images painted or engraved on cave walls?", "cave art"),
    ("number of honeybee workers in a hive", "workers", "What reproductive role does a male honeybee have in a colony?", "mating with a queen"),
    ("number of computer passwords tested", "passwords", "What security process confirms a user's identity before granting access?", "authentication"),
    ("length of a solar calendar used by a community", "days", "What extra day is added to February in a leap year in the Gregorian calendar?", "February 29"),
    ("mass of clay used for a sculpture", "kg", "What sculpting process builds a shape by adding and shaping a soft material such as clay?", "modeling"),
    ("number of tickets sold for a concert", "tickets", "What is a group of musicians performing together called?", "ensemble"),
    ("number of cells examined under a microscope", "cells", "What part of a eukaryotic cell contains most of its DNA?", "nucleus"),
    ("length of a section of railway track", "m", "What is the distance between the inner sides of a railway's two rails called?", "track gauge"),
    ("number of coins in a currency collection", "coins", "What is the study and collection of coins and currency called?", "numismatics"),
    ("number of letters in an encoded text", "letters", "What is a cipher that shifts each letter by a fixed number of positions in the alphabet called?", "Caesar cipher"),
    ("volume of milk used for fermented food", "L", "What type of bacteria convert milk sugar into lactic acid during yogurt production?", "lactic acid bacteria"),
    ("number of people using a public bus", "people", "What term describes transportation services shared by passengers and available to the public?", "public transit"),
    ("length of a courtroom hearing", "hours", "What sworn statement is given by a witness in court?", "testimony"),
    ("mass of recycled paper used in a workshop", "kg", "What process turns used paper into pulp so it can be made into new paper?", "paper recycling"),
    ("number of matches played in a volleyball tournament", "matches", "What volleyball player wears a different-colored jersey and specializes in defensive play?", "libero"),
    ("number of objects in a digital database", "objects", "What database language is commonly used to query relational tables?", "SQL"),
    ("length of thread used to bind a book", "m", "What process joins the pages of a book together within a cover?", "bookbinding"),
    ("number of paintings in a gallery", "paintings", "What is a painting of a person's face or likeness called?", "portrait"),
    ("number of specimens in a bird migration study", "birds", "What technique attaches a numbered band to a bird's leg to track its movements?", "bird banding"),
    ("mass of a soap batch", "kg", "What reaction produces soap from fats or oils and an alkali?", "saponification"),
    ("number of containers used to sort household waste", "containers", "What process collects and processes discarded material into new products?", "recycling"),
    ("number of lines in a geometry exercise", "lines", "What are two lines in a plane that never intersect called?", "parallel lines"),
    ("number of photographs in a landscape series", "photographs", "What camera setting controls the size of the opening through which light enters?", "aperture"),
    ("number of sculptures in a public garden", "sculptures", "What sculpting process removes material from stone or wood to make a shape?", "carving"),
    ("length of rope in a climbing equipment bag", "m", "What piece of climbing equipment is a metal loop with a spring-loaded gate?", "carabiner"),
    ("number of moves played in a chess game", "moves", "What chess piece moves in an L-shaped pattern?", "knight"),
    ("number of boats in a sailing race", "boats", "What device on a sailboat is adjusted to change the angle of a sail to the wind?", "sheet"),
    ("number of words in a translated sentence", "words", "What term describes a word that has the opposite meaning of another word?", "antonym"),
    ("volume of fruit juice in a container", "L", "What process uses heat to reduce harmful microorganisms in juice without sterilizing it?", "pasteurization"),
    ("number of stars observed in a galaxy survey", "stars", "What is the name for a star that suddenly increases greatly in brightness because of a stellar explosion?", "supernova"),
    ("number of garments made from fabric", "garments", "What fastening device joins two edges of fabric using interlocking teeth?", "zipper"),
    ("mass of ground coffee used in a brewing trial", "g", "What coffee brewing method forces hot water through finely ground coffee under pressure?", "espresso"),
    ("number of rows in a spreadsheet", "rows", "What spreadsheet expression performs a calculation using cell values?", "formula"),
    ("number of books in a classroom", "books", "What form of writing tells a story using invented characters and events?", "fiction"),
    ("length of a television program", "minutes", "What term describes text displayed on screen to represent spoken dialogue?", "subtitles"),
    ("number of drawings of human organs in an anatomy study", "drawings", "What human organ filters blood and produces urine?", "kidney"),
    ("number of shapes in a geometry diagram", "shapes", "What is a three-dimensional solid with six square faces called?", "cube"),
    ("number of metals tested for their ability to be drawn into wire", "metals", "What property allows a metal to be drawn into wire without breaking?", "ductility"),
    ("number of plants raised for a school garden", "plants", "What gardening practice moves a young plant from one location to another?", "transplanting"),
    ("mass of paint used for a large mural", "kg", "What color results from mixing blue and yellow pigments?", "green"),
    ("number of items in a restaurant menu", "items", "What course is traditionally served after the main course of a meal?", "dessert"),
    ("number of calls received at an emergency dispatch center", "calls", "What emergency service is primarily responsible for extinguishing fires?", "fire service"),
    ("number of seeds sown in a field", "seeds", "What agricultural practice supplies nutrients to soil using compost or other amendments?", "fertilization"),
    ("number of measurements in a weather report", "measurements", "What term describes the temperature at which water vapor in air begins to condense?", "dew point"),
    ("number of folders in a computer file system", "folders", "What is the topmost directory in a hierarchical file system called?", "root directory"),
]


def main() -> None:
    for path in (OUTPUT, SCORES):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    if source["candidate_count"] != 208 or len(source["blocks"]) != 208:
        raise ValueError("Expected the 208-block draft")
    if len(TOPICS) != 84:
        raise ValueError(f"Expected 84 topic definitions, got {len(TOPICS)}")
    originals = source["blocks"]
    instructions = {
        task_type: next(b["user_instruction"] for b in originals if b["task_type"] == task_type)
        for task_type in ("date_location_extraction", "target_quantity_extraction")
    }
    new = []
    for index in range(42):
        task_type = "date_location_extraction" if index < 21 else "target_quantity_extraction"
        block = new_block(409 + index, index, task_type, instructions[task_type], topics=TOPICS)
        block["plan_version"] = "batch_05_v1"
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
    old_answers = {
        probe["expected_answer"].casefold()
        for block in originals for probe in block["probes"].values()
    }
    new_answers = set()
    for block in new:
        clean = " ".join(
            [block["user_instruction"]] + [block["contexts"][label]["text"] for label in "AB"]
        ).casefold()
        for probe in block["probes"].values():
            answer = probe["expected_answer"].casefold()
            if answer in clean:
                raise ValueError(f"{block['block_id']}: expected answer appears in clean text")
            if answer in old_answers or answer in new_answers:
                raise ValueError(f"{block['block_id']}: duplicate expected answer {answer}")
            new_answers.add(answer)
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
        "collection_id": "pilot_expansion_250_draft_v1",
        "status": "count_complete_draft_pending_human_review",
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
