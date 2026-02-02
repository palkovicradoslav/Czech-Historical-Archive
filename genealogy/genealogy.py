import json
from datetime import datetime
from jarowinkler import jarowinkler_similarity
import re
from unidecode import unidecode
import unicodedata
import os
from collections import defaultdict
import math


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

RECORDS_DIR = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'structured_records'))

GEN_RECORDS_DIR = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'pages', 'genealogy_structured_records'))

# Minimum plausible generation gap (parent → child)
MIN_PARENT_MARRIAGE_AGE = 14
# Defensive approach to avoid improbable parent ages
MAX_PARENT_MARRIAGE_AGE = 50
# Fuzzy-match threshold for considering two names identical
NAME_SIMILARITY_THRESHOLD = 0.9
# Probabilistic match threshold
MATCH_PROBABILITY_THRESHOLD = 0.85


def parse_date(datestr):
    """Parse YYYY-MM-DD or return None if invalid."""
    try:
        return datetime.strptime(datestr, "%Y-%m-%d").date()
    except Exception:
        return None


def sequence_similarity(a, b):
    """Compare and quantify similarity of two strings."""
    if not a or not b:
        return 0.0
    return jarowinkler_similarity(a.lower(), b.lower())


def generate_surname_versions(surname):
    """Generate possible variations of a surname based on Czech naming patterns."""
    if not is_valid(surname):
        return set()

    surname = surname.strip().lower()
    possible_surnames = {surname}

    RULES = [
        ("cova", lambda base: {base + "ec", base + "c"}),
        ("kova", lambda base: {base + "ek", base + "k"}),
        ("ska", lambda base: {base + "sky", base + "sk"}),
        ("cka", lambda base: {base + "cky", base + "ck"}),
        ("na", lambda base: {base + "ny", base + "n"}),
        ("ta", lambda base: {base + "ty", base + "t"}),
        ("a", lambda base: {base, base + "y"}
         if not surname.endswith("ova") else set()),
        ("ova", lambda base: {base, base + "a"}),
    ]

    for suffix, transform in RULES:
        if surname.endswith(suffix):
            base = surname[: -len(suffix)]
            possible_surnames.update(transform(base))

    # Extra: vowel stems for -ova
    if surname.endswith("ova") and len(surname) > 4:
        base = surname[:-3]
        if base.endswith(("i", "u")):
            possible_surnames.add(base)

        # consonant doubling, e.g., "kossova" → "kos"
        if len(base) >= 2 and base[-1] == base[-2]:
            possible_surnames.add(base[:-1])

    possible_surnames.discard("")
    return possible_surnames


def surname_similarity(a, b):
    """Check if two surnames are similar considering surname variations."""
    try:
        a = a.strip().split()[-1]
        b = b.strip().split()[-1]
    except IndexError:
        return False

    surnames = generate_surname_versions(a)

    for p_surname in surnames:
        similarity = sequence_similarity(p_surname, b)
        if similarity >= NAME_SIMILARITY_THRESHOLD:
            return True

    surnames = generate_surname_versions(b)

    for p_surname in surnames:
        similarity = sequence_similarity(p_surname, a)
        if similarity >= NAME_SIMILARITY_THRESHOLD:
            return True

    return False


def full_name_similarity(a, b):
    """Check if two full names are similar considering both first names and surnames."""
    try:
        first_names_a = " ".join(a.strip().split()[:-1])
        first_names_b = " ".join(b.strip().split()[:-1])
    except IndexError:
        return False

    return surname_similarity(a, b) and sequence_similarity(first_names_a, first_names_b) >= NAME_SIMILARITY_THRESHOLD


def is_valid(value):
    return value is not None and value != "null"


class Person:
    _next_id = 0

    def __init__(self, full_name):
        self.id = Person._next_id
        Person._next_id += 1
        self.full_name = full_name
        self.first_name, self.surname = self._split_name(full_name)
        self.maiden_name = None
        self.birthdate = None
        self.deathdate = None
        self.weddingdate = None
        self.birthplace = None
        self.deathplace = None
        self.father = None
        self.mother = None
        self.spouses = set()    # set of Person
        self.children = set()   # set of Person

    def _split_name(self, full):
        """Split a full name into first name and surname."""
        if not full:
            return "", ""
        parts = full.strip().split()
        if len(parts) >= 2:
            return " ".join(parts[:-1]), parts[-1]
        return full, ""

    def __repr__(self):
        return f"<Person id={self.id} name={self.full_name!r} place={self.birthplace} birthdate={self.birthdate} maiden={self.maiden_name}>"


class FamilyTreeBuilder:
    def __init__(self):
        # id → Person
        self.people = {}
        # surname -> person_id
        self._surname_index = {}
        # name frequency tracking for probabilistic weighting
        self._name_frequencies = defaultdict(int)
        self._total_names = 0

    def _register(self, person):
        """Register a person in the family tree and update surname indices."""
        self.people[person.id] = person

        # Add to the new index
        normalized_surname = self._clean_name(person.surname)
        if normalized_surname not in self._surname_index:
            self._surname_index[normalized_surname] = set()
        self._surname_index[normalized_surname].add(person.id)

        # Track name frequencies for probabilistic scoring
        self._name_frequencies[self._clean_name(person.full_name)] += 1
        self._total_names += 1

        return person

    def _add_child(self, person, child):
        """Add a child to a person's children."""
        if child not in person.children:
            person.children.add(child)

    def _same_birthday(self, p, birthdate, mismatches_allowed=0):
        """Check if two birthdates are the same, allowing for one character mismatch."""
        if p.birthdate and is_valid(birthdate):
            bd1 = p.birthdate.strftime("%Y-%m-%d")
            bd2 = birthdate.strftime("%Y-%m-%d")

            mismatch_count = sum(a != b for a, b in zip(bd1, bd2))

            return mismatch_count <= mismatches_allowed

        # Cannot determine, assume match
        return True

    def _clean_name(self, s):
        """Remove diacritics and convert to lowercase."""
        if not s:
            return ""
        return unidecode(unicodedata.normalize('NFD', s.lower()))

    def _same_place(self, person, place_of_living, scale=1.0):
        """Check if a person's birthplace matches the given place of living."""
        if not person.birthplace or not is_valid(place_of_living):
            return True

        # Both places are present; perform a similarity check.
        cleaned_birthplace = self._clean_name(person.birthplace)
        cleaned_place_of_living = self._clean_name(place_of_living)

        return sequence_similarity(cleaned_birthplace, cleaned_place_of_living) >= NAME_SIMILARITY_THRESHOLD * scale

    def _same_parent(self, person, parent_name_to_match, mother=False, scale=1.0):
        """Verify that a parent's name matches the given name."""
        if not person or not is_valid(parent_name_to_match):
            return True

        cleaned_person_name = self._clean_name(person.full_name)
        cleaned_parent_name_to_match = self._clean_name(parent_name_to_match)
        if sequence_similarity(cleaned_person_name, cleaned_parent_name_to_match) >= NAME_SIMILARITY_THRESHOLD * scale:
            return True

        if mother:
            full_maiden_name = self._get_full_maiden_name(
                person.full_name, person.maiden_name)
            if sequence_similarity(full_maiden_name, cleaned_parent_name_to_match) >= NAME_SIMILARITY_THRESHOLD * scale:
                return True

        return False

    def _get_full_maiden_name(self, name, maiden_name):
        """Get cleaned maiden name."""
        if not is_valid(name):
            return ""
        if is_valid(maiden_name):
            return self._clean_name(" ".join(name.split()[:-1] + [maiden_name]))
        return self._clean_name(name)

    def _child_gap(self, p, birthdate, death_date):
        """Verify that children's birthdates are compatible with parent's birth and death dates."""
        for child in p.children:
            if is_valid(birthdate) and is_valid(child.birthdate):
                gap = abs(child.birthdate.year - birthdate.year)
                if gap < MIN_PARENT_MARRIAGE_AGE or gap > MAX_PARENT_MARRIAGE_AGE:
                    return False
            if is_valid(death_date) and is_valid(child.birthdate):
                if child.birthdate.year > death_date.year:
                    return False
        return True

    def _possible_date(self, date1, date2, comparator=lambda a, b: a <= b, gap=0):
        """Compute if dates are compatible."""
        if date1 is not None and date2 is not None:
            return comparator(date1.year + gap, date2.year)
        return True

    def _calculate_highest_name_similarity(self, person, name_to_match, maiden_name_to_match, father_derived_name_to_match):
        """Compute name similarity based on full name, maiden name, and father-derived name."""
        canon_name = self._clean_name(person.full_name)
        canon_maiden_name = self._get_full_maiden_name(
            person.full_name, person.maiden_name)

        name_sim = sequence_similarity(canon_name, name_to_match)
        maiden_name_sim = sequence_similarity(
            canon_maiden_name, maiden_name_to_match) if maiden_name_to_match else 0

        father_derived_sim = 0
        if father_derived_name_to_match:
            father_derived_sim = sequence_similarity(
                canon_name, father_derived_name_to_match)

        return max(name_sim, maiden_name_sim, father_derived_sim)

    def _calculate_probabilistic_score(self, person, name, birthdate, place_of_living,
                                       father_name, mother_name, maiden_name):
        """Calculate a probabilistic match score inspired by Fellegi-Sunter model."""
        log_likelihood = 0.0

        # Name similarity with frequency-based weighting
        cleaned_name = self._clean_name(name)
        name_freq = self._name_frequencies.get(cleaned_name, 1)
        u_prob = max(name_freq / self._total_names,
                     0.001) if self._total_names > 0 else 0.01

        canon_name = self._clean_name(person.full_name)
        canon_maiden_name = self._get_full_maiden_name(
            person.full_name, person.maiden_name)

        name_sim = sequence_similarity(canon_name, cleaned_name)
        if is_valid(maiden_name):
            cleaned_maiden_name = self._get_full_maiden_name(name, maiden_name)
            maiden_sim = sequence_similarity(
                canon_maiden_name, cleaned_maiden_name)
            name_sim = max(name_sim, maiden_sim)

        if name_sim >= NAME_SIMILARITY_THRESHOLD:
            # Rare names get higher weight
            bayes_factor = 0.95 / u_prob
            log_likelihood += math.log(bayes_factor)
        else:
            # Partial name match gets reduced weight
            bayes_factor = 0.7 / u_prob
            log_likelihood += math.log(bayes_factor) * 0.5

        # Birthdate match
        if is_valid(birthdate) and person.birthdate:
            if self._same_birthday(person, birthdate, mismatches_allowed=0):
                # Exact birthdate match: m=0.98, u=1/36500 (100 years)
                log_likelihood += math.log(0.98 / (1/36500))
            else:
                # Birthdate mismatch is strong negative evidence
                log_likelihood += math.log(0.02 / (36499/36500))

        # Place similarity
        if is_valid(place_of_living) and person.birthplace:
            place_sim = sequence_similarity(
                self._clean_name(person.birthplace),
                self._clean_name(place_of_living)
            )
            if place_sim >= NAME_SIMILARITY_THRESHOLD:
                # m=0.85, u=0.05 (considering 20 common places)
                log_likelihood += math.log(NAME_SIMILARITY_THRESHOLD / 0.05)
            else:
                log_likelihood += math.log((1 -
                                           NAME_SIMILARITY_THRESHOLD) / 0.95)

        # Parent names
        parent_matches = 0
        parent_checks = 0

        if is_valid(father_name) and person.father:
            parent_checks += 1
            if self._same_parent(person.father, father_name):
                parent_matches += 1

        if is_valid(mother_name) and person.mother:
            parent_checks += 1
            if self._same_parent(person.mother, mother_name, mother=True):
                parent_matches += 1

        if parent_checks > 0:
            if parent_matches == parent_checks:
                # All parents match: m=0.95, u=0.01
                log_likelihood += math.log(0.95 / 0.01) * parent_matches
            elif parent_matches > 0:
                # Partial parent match
                log_likelihood += math.log(0.3 / 0.1) * parent_matches
            else:
                # No parent matches despite having parent info
                log_likelihood += math.log(0.05 / 0.99) * parent_checks

        # Convert log-likelihood to probability using logistic function
        # assuming 1% of candidates are true matches
        prior_odds = 0.01 / 0.99
        posterior_odds = prior_odds * math.exp(log_likelihood)
        probability = posterior_odds / (1 + posterior_odds)

        return probability

    def _passes_detailed_checks(self, person, birthdate, death_date, place_of_living,
                                father_name, mother_name, date_blocker):
        """Decide if a person passes detailed checks based on provided attributes"""
        if not self._same_birthday(person, birthdate, mismatches_allowed=1):
            return False
        if not self._same_place(person, place_of_living, scale=0.95):
            return False
        if not self._same_parent(person.father, father_name, scale=0.95):
            return False
        if not self._same_parent(person.mother, mother_name, mother=True, scale=0.95):
            return False
        if not self._child_gap(person, birthdate, death_date):
            return False

        # A parent must be old enough at the time of the event - if provided
        if not self._possible_date(person.birthdate, date_blocker, gap=MIN_PARENT_MARRIAGE_AGE):
            return False
        # Also cannot have died
        if not self._possible_date(person.deathdate, date_blocker, comparator=lambda a, b: a >= b):
            return False
        # Cannot have been born after death
        if not self._possible_date(person.birthdate, date_blocker, comparator=lambda a, b: a <= b):
            return False
        return True

    def _get_candidates_by_surname(self, name):
        """Generate candidate person IDs based on surname variations."""
        if not is_valid(name):
            return set()

        surname_to_check = self._clean_name(name.split()[-1])
        candidate_ids = self._surname_index.get(surname_to_check, set())

        for variation in generate_surname_versions(surname_to_check):
            candidate_ids.update(self._surname_index.get(variation, set()))

        return candidate_ids

    def _find_or_create(self, name, birthdate=None, death_date=None, place_of_living=None,
                        father_name=None, mother_name=None, maiden_name=None,
                        date_blocker=None, family=None, wedding_date=None):
        """Finds the best matching person or creates a new one using probabilistic scoring. """

        # clean and generate name variations for matching
        cleaned_name = self._clean_name(name)
        cleaned_maiden_name = self._get_full_maiden_name(name, maiden_name)
        father_derived_maiden_name = None
        if is_valid(father_name):
            father_surname = father_name.split()[-1]
            father_derived_maiden_name = self._get_full_maiden_name(
                name, father_surname)

        identified_candidates = []
        # check if enough information is available for thorough search
        is_detailed_search = any(arg is not None for arg in [
                                 birthdate, place_of_living, father_name, mother_name])

        # Get a smaller list of candidates from the index first
        candidate_ids = self._get_candidates_by_surname(name)
        candidate_ids.update(
            self._get_candidates_by_surname(cleaned_maiden_name))
        candidate_ids.update(self._get_candidates_by_surname(
            father_derived_maiden_name))

        for person_id in candidate_ids:
            p = self.people[person_id]
            # avoid self-references
            if p == family:
                continue

            # check if old enough
            if family and family.birthdate and p.birthdate and \
               (family.birthdate.year - p.birthdate.year) < MIN_PARENT_MARRIAGE_AGE:
                continue

            # name similarity check before expensive probabilistic scoring
            similarity_score = self._calculate_highest_name_similarity(
                p, cleaned_name, cleaned_maiden_name, father_derived_maiden_name)

            if similarity_score < NAME_SIMILARITY_THRESHOLD:
                continue

            # apply stricter filtering with hard constraints
            if is_detailed_search and not self._passes_detailed_checks(p, birthdate, death_date,
                                                                       place_of_living, father_name,
                                                                       mother_name, date_blocker):
                continue

            # check children born before wedding date
            if wedding_date and p.children:
                has_children_before_wedding = any(
                    child.birthdate and child.birthdate < wedding_date
                    for child in p.children)
                if has_children_before_wedding:
                    continue

            # Calculate probabilistic score for candidates that pass hard constraints
            prob_score = self._calculate_probabilistic_score(
                p, name, birthdate, place_of_living,
                father_name, mother_name, maiden_name
            )

            # Only consider candidates with sufficient probability
            if prob_score >= MATCH_PROBABILITY_THRESHOLD:
                identified_candidates.append((p, prob_score))

        # return the best match based on probabilistic score
        if identified_candidates:
            best_match, _ = max(identified_candidates,
                                key=lambda item: item[1])
            self._update_person(best_match,
                                birthdate=birthdate,
                                place_of_living=place_of_living,
                                maiden_name=maiden_name)
            return best_match

        # no suitable person found so create a new one
        new_person = Person(name)
        if is_valid(birthdate):
            new_person.birthdate = birthdate
        if is_valid(place_of_living):
            new_person.birthplace = place_of_living
        if is_valid(maiden_name):
            new_person.maiden_name = maiden_name
        if is_valid(death_date):
            new_person.deathdate = death_date

        return self._register(new_person)

    def _update_person(self, person, birthdate=None, place_of_living=None, name=None, maiden_name=None):
        """Update person's information."""
        if person.birthdate is None and is_valid(birthdate):
            person.birthdate = birthdate
        if person.birthplace is None and is_valid(place_of_living):
            person.birthplace = place_of_living

        if is_valid(name):
            person.full_name = name
        if person.maiden_name is None and is_valid(maiden_name):
            person.maiden_name = maiden_name

    def _get_last_name(self, name):
        """Extract the last name from a full name string."""
        if name is None:
            return None
        names = name.split()
        return names[-1].strip() if len(names) > 0 else None

    def _get_married_name(self, wife, husband):
        """Generate a wife's married name using husband's surname."""
        if husband == "" or wife == "":
            return wife
        wife_names = wife.split()
        family_name = self._get_last_name(husband)
        if family_name != "" and family_name is not None:
            return " ".join(wife_names[:-1] + [family_name])
        return wife

    def _link_parents(self, p, father_name, mother_name, father_place=None, mother_place=None):
        """Link parents to a person."""
        mother = None
        father = None
        if is_valid(father_name):
            # find the parent
            father = self._find_or_create(father_name,
                                          place_of_living=father_place,
                                          date_blocker=p.birthdate if p is not None else None,
                                          family=p)
            if p.father is None:
                p.father = father
                # connect the parent - child link
            self._update_person(father, place_of_living=father_place)
            self._add_child(father, p)

        num_of_people = len(self.people)
        if is_valid(mother_name):
            married_name = self._get_married_name(
                mother_name, father.full_name if father is not None else "")
            maiden_name = self._get_last_name(mother_name)
            mother = self._find_or_create(mother_name,
                                          place_of_living=mother_place,
                                          maiden_name=maiden_name,
                                          date_blocker=p.birthdate if p is not None else None,
                                          family=p)
            if p.mother is None:
                p.mother = mother

            # connect the parent - child link
            self._update_person(mother, place_of_living=mother_place,
                                name=married_name, maiden_name=maiden_name)
            self._add_child(mother, p)

        self._connect_married_parents(mother, father, num_of_people)

    def _are_plausible_spouses(self, person1, person2):
        """Check if two people could plausibly be married based on their ages and dates."""
        if not person1 or not person2:
            return False
        if not self._possible_date(person1.birthdate, person2.deathdate, gap=MIN_PARENT_MARRIAGE_AGE):
            return False
        if not self._possible_date(person2.birthdate, person1.deathdate, gap=MIN_PARENT_MARRIAGE_AGE):
            return False

        return True

    def _have_matching_surnames_for_marriage(self, father, mother):
        """Check if marriage and maiden surnames match for a potential person connection."""
        if not father or not mother:
            return False

        father_surname = self._clean_name(father.full_name.split()[-1])
        mother_surname = self._clean_name(mother.full_name.split()[-1])
        mother_maiden_name = self._clean_name(
            mother.maiden_name) if mother.maiden_name else ""

        # both maiden and married surnames can match
        if sequence_similarity(father_surname, mother_surname) >= NAME_SIMILARITY_THRESHOLD:
            return True

        if mother_maiden_name and sequence_similarity(father_surname, mother_maiden_name) >= NAME_SIMILARITY_THRESHOLD:
            return True

        return False

    def _connect_married_parents(self, mother, father, people_count_before_ingest):
        """Connect parents as spouses if they meet the marriage criteria."""
        both_parents_existed = (people_count_before_ingest == len(self.people))

        if not mother or not father or not both_parents_existed:
            return

        if self._are_plausible_spouses(mother, father) and self._have_matching_surnames_for_marriage(father, mother):
            father.spouses.add(mother)
            mother.spouses.add(father)

    def ingest_birth(self, rec):
        """Process a birth record and create or update person and relationships."""
        # processing birth without name makes no sense
        if not is_valid(rec.get("name")):
            return

        bd = parse_date(rec.get("birthdate", ""))

        # If surname not provided in name, try to infer from father
        child_name = f"{rec.get('name', '').strip()} {rec.get('surname', '').strip()}"
        if len(child_name.split()) <= 1 and rec.get('father'):
            father_name = rec.get('father')
            father_surname = father_name.split(
            )[-1] if ' ' in father_name else None
            if father_surname:
                child_name = f"{child_name} {father_surname}"

        birthplace = rec.get("birthplace", None)
        p = self._find_or_create(child_name,
                                 birthdate=bd,
                                 place_of_living=birthplace,
                                 father_name=rec.get("father"),
                                 mother_name=rec.get("mother"))

        if is_valid(bd):
            p.birthdate = bd

        if is_valid(birthplace):
            p.birthplace = birthplace

        # link parents
        self._link_parents(p, rec.get("father"), rec.get("mother"), rec.get(
            "father_place_of_living"), rec.get("mother_place_of_living"))

        return p

    def ingest_marriage(self, rec):
        """Process a marriage record and create or update persons and relationships."""
        # processing marriage without names makes no sense
        if not is_valid(rec.get("groom_surname")) or not is_valid(rec.get("bride_surname")):
            return

        # dates
        wed = parse_date(rec.get("wedding_date", None))

        # birthdates
        g_bd = parse_date(rec.get("groom_birthday", None))
        b_bd = parse_date(rec.get("bride_birthday", None))

        groom = self._find_or_create(f"{rec['groom_name']} {rec['groom_surname']}",
                                     birthdate=g_bd,
                                     father_name=rec.get("groom_father"),
                                     mother_name=rec.get("groom_mother"),
                                     date_blocker=wed,
                                     wedding_date=wed)
        bride = self._find_or_create(f"{rec['bride_name']} {rec['groom_surname']}",
                                     maiden_name=rec['bride_surname'],
                                     birthdate=b_bd,
                                     father_name=rec.get("bride_father"),
                                     mother_name=rec.get("bride_mother"),
                                     date_blocker=wed,
                                     wedding_date=wed)

        if is_valid(wed):
            groom.weddingdate = wed
            bride.weddingdate = wed

        self._update_person(groom, birthdate=g_bd)
        self._update_person(bride, birthdate=b_bd)

        # parents
        self._link_parents(groom, rec.get("groom_father"), rec.get(
            "groom_mother"), rec.get("groom_father_place"), rec.get("groom_mother_place"))
        self._link_parents(bride, rec.get("bride_father"), rec.get(
            "bride_mother"), rec.get("bride_father_place"), rec.get("bride_mother_place"))

        # link spouses
        groom.spouses.add(bride)
        bride.spouses.add(groom)

        return groom, bride

    def ingest_death(self, rec):
        """Process a death record and update person's death information and relationships."""
        name = rec.get("name")
        if not is_valid(name):
            return

        dd = parse_date(rec.get("date_of_death", ""))
        additional_info = rec.get('additional_info', '')
        birthdate = None
        deathplace = None

        if is_valid(additional_info):
            birthdate = re.search(
                r"(?<=\bborn on\s)\d{4}-\d{2}-\d{2}\b", additional_info)
            birthdate = parse_date(birthdate.group(0)) if birthdate else None
            # birthdate must be before death
            birthdate = birthdate if birthdate and (
                dd is None or birthdate <= dd) else None

            deathplace = re.search(
                r"Buried in\s+([^;]+)", additional_info, re.IGNORECASE)
            deathplace = deathplace.group(1) if deathplace else None

        # Process parent information
        father_name = rec.get("father", None)
        mother_name = rec.get("mother", None)

        p = self._find_or_create(name,
                                 birthdate=birthdate,
                                 death_date=dd,
                                 place_of_living=rec.get(
                                     "place_of_living", None),
                                 father_name=father_name,
                                 mother_name=mother_name)
        if is_valid(dd):
            p.deathdate = dd
        if is_valid(deathplace):
            p.deathplace = deathplace

        self._link_parents(p, rec.get("father"), rec.get("mother"), rec.get(
            "father_place_of_living"), rec.get("mother_place_of_living"))

        new_maiden_name = None
        if p.full_name != name and not p.maiden_name and p.father:
            new_maiden_name = p.father.full_name.split()[-1]

        self._update_person(p,
                            birthdate=birthdate,
                            place_of_living=rec.get("place_of_living"),
                            name=(
                                name if p.full_name != name and new_maiden_name is not None else p.full_name),
                            maiden_name=new_maiden_name)

        return p


def process_and_save(records_dir, out_dir):
    """Process records from input directory and save enriched records to output directory."""
    os.makedirs(out_dir, exist_ok=True)
    builder = FamilyTreeBuilder()

    order = ['birth', 'death', 'marriage']

    # Process files one by one in order
    for record_type in order:
        for fname in os.listdir(records_dir):
            if not fname.lower().endswith('.json'):
                continue
            if record_type not in fname.lower():
                continue

            in_path = os.path.join(records_dir, fname)
            with open(in_path, 'r', encoding='utf-8') as fr:
                records = json.load(fr)
            out_path = os.path.join(out_dir, fname)
            enriched = {}
            # ingest each record into the builder
            for key, rec in records.items():
                typ = rec.get("record_type")
                if typ == 'birth':
                    p = builder.ingest_birth(rec)
                    if p is None:
                        continue
                    rec['person_id'] = str(p.id)
                elif typ == 'marriage':
                    people = builder.ingest_marriage(rec)
                    if people is None:
                        continue
                    groom, bride = people
                    rec['groom_id'] = str(groom.id)
                    rec['bride_id'] = str(bride.id)
                elif typ == 'death':
                    p = builder.ingest_death(rec)
                    if p is None:
                        continue
                    rec['person_id'] = str(p.id)
                else:
                    # unknown record type
                    continue
                enriched[key] = rec
            # Save each enriched record list back to its own file
            with open(out_path, 'w', encoding='utf-8') as fw:
                json.dump(enriched, fw, indent=2, ensure_ascii=False)
            print(f"Processed and saved: {fname}")

    return builder
