"""Curated IPC -> BNS mapping for the 100 ILSI labels, plus BNS offences with no IPC equivalent.

Every mapping row was checked by reading the cited source row. Conventions:
  verified = "yes"  the cited row states this mapping explicitly
  verified = "no"   our inference (sub-clause level, amount-dependent split, or the IPC
                    section is not listed in the source); see notes
  mapping_type:
    one_to_one   one BNS (sub-)section; wording/punishment changes only (noted)
    one_to_many  one IPC section split over several BNS sub-sections
    many_to_one  several ILSI labels land on the same BNS section
    modified     elements/scope of the offence changed, so the same facts may not map the same way
    repealed     IPC section repealed before BNS; no BNS counterpart
    unclear      no verifiable BNS counterpart; mapping withheld
  ipc_title: standard IPC marginal headings (they match the source's subject column where
             visible); titles were not separately verified and are not legal mappings.

Source caveat: no official MHA/BPR&D correspondence table could be accessed, so "verified"
means verified against the CAPT Bhopal table cited in `source` (a secondary source).

Usage: python src/bns_mapping.py
"""
import argparse
import json

import pandas as pd

from common import PROC

CAPT = ("CAPT Bhopal correspondence table (A.K. Yadav, IPS, Director, CAPT Bhopal), "
        "https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf "
        "[secondary source; not the MHA/BPR&D official table]")
ILSI = "ILSI statute text (Zenodo 6053791, secs.jsonl)"

# ipc: (ipc_title, bns_sections, mapping_type, verified, notes)
ROWS = {
    "2": ("Punishment of offences committed within India", "1(3)", "one_to_one", "yes", "Application provision, not an offence; folded into BNS s.1 as a sub-section."),
    "3": ("Punishment of offences committed beyond, but which by law may be tried within, India", "1(4)", "one_to_one", "yes", "Application provision; 'Indian laws' -> 'law ... for the time being in force in India'."),
    "4": ("Extension of Code to extra-territorial offences", "1(5)", "one_to_one", "yes", "Application provision; illustration 'Uganda' -> 'any place outside India'."),
    "5": ("Certain laws not to be affected by this Act", "1(6)", "one_to_one", "yes", "Application provision; included as sub-section."),
    "13": ("[Repealed] Definition of 'Queen'", "", "repealed", "yes", "ILSI statute text reads 'Rep. by the A.O. 1950.' No BNS counterpart. Its presence as a label in modern cases is likely label noise. (Title from general knowledge, unverified.)"),
    "34": ("Acts done by several persons in furtherance of common intention", "3(5)", "one_to_one", "yes", "Included as sub-section of BNS 3 (general explanations); no other change."),
    "107": ("Abetment of a thing", "45", "one_to_one", "yes", "'Court of justice' -> 'Court'."),
    "109": ("Punishment of abetment if the act abetted is committed in consequence and where no express provision is made for its punishment", "49", "one_to_one", "yes", "No change."),
    "114": ("Abettor present when offence is committed", "54", "one_to_one", "yes", "No change."),
    "120": ("Concealing design to commit offence punishable with imprisonment", "60", "one_to_one", "yes", "No change."),
    "120B": ("Punishment of criminal conspiracy", "61(2)", "one_to_one", "yes", "Included as sub-section; definition (IPC 120A) is 61(1)."),
    "143": ("Punishment (for being member of unlawful assembly)", "", "unclear", "no", "Mapping withheld: IPC 143 is not listed in the CAPT table (it maps IPC 142 to BNS 189(2)), so no BNS section could be verified."),
    "147": ("Punishment for rioting", "191(2)", "one_to_one", "yes", "Included as sub-section sans heading."),
    "148": ("Rioting, armed with deadly weapon", "191(3)", "one_to_one", "yes", "Max imprisonment raised from 3 to 5 years."),
    "149": ("Every member of unlawful assembly guilty of offence committed in prosecution of common object", "190", "one_to_one", "yes", "No change."),
    "155": ("Liability of person for whose benefit riot is committed", "193(2)", "one_to_one", "yes", "Included as sub-section."),
    "156": ("Liability of agent of owner or occupier for whose benefit riot is committed", "193(3)", "one_to_one", "yes", "'Ditto' (included as sub-section)."),
    "161": ("[Repealed] Public servant taking gratification", "", "repealed", "yes", "ILSI statute text: 'Rep. by the Prevention of Corruption Act, 1988 (49 of 1988), sec. 31.' Offence now under the PC Act, not BNS."),
    "164": ("[Repealed] Punishment for abetting offence defined in section 162 or 163 by public servant", "", "repealed", "yes", "ILSI statute text: 'Rep. by the Prevention of Corruption Act, 1988 (49 of 1988), sec. 31.' (Title from general knowledge, unverified.)"),
    "173": ("Preventing service of summons or other proceeding, or preventing publication thereof", "207", "one_to_one", "yes", "'Ditto' in table."),
    "174A": ("Non-appearance in response to a proclamation under section 82 of Act 2 of 1974", "209", "one_to_one", "yes", "Now refers to s.84 BNSS; community service added as punishment."),
    "186": ("Obstructing public servant in discharge of public functions", "221", "one_to_one", "yes", "Max fine raised from Rs 500 to Rs 2,500."),
    "188": ("Disobedience to order duly promulgated by public servant", "223", "one_to_one", "yes", "Reorganised as clauses (a)/(b); punishments increased."),
    "190": ("Threat of injury to induce person to refrain from applying for protection to public servant", "225", "one_to_one", "yes", "No change."),
    "193": ("Punishment for false evidence", "229", "one_to_one", "yes", "Fines specified (Rs 10,000 / Rs 5,000)."),
    "200": ("Using as true such declaration knowing it to be false", "237", "one_to_one", "yes", "Formal change in explanation."),
    "201": ("Causing disappearance of evidence of offence, or giving false information to screen offender", "238", "one_to_one", "yes", "Clauses named (a), (b), (c)."),
    "228": ("Intentional insult or interruption to public servant sitting in judicial proceeding", "267", "one_to_one", "yes", "Max fine raised from Rs 1,000 to Rs 5,000."),
    "229A": ("Failure by person released on bail or bond to appear in Court", "269", "one_to_one", "yes", "No change."),
    "279": ("Rash driving or riding on a public way", "281", "one_to_one", "yes", "No change."),
    "294": ("Obscene acts and songs", "296", "one_to_one", "yes", "Fine specified (up to Rs 1,000). The clause-level label 294(b) has its mapping withheld."),
    "294(b)": ("Obscene acts and songs: clause (b), obscene songs/words in or near a public place", "", "unclear", "no", "Mapping withheld: the table confirms IPC 294 -> BNS 296 only at section level; the clause-level correspondence could not be verified."),
    "299": ("Culpable homicide", "100", "one_to_one", "yes", "No change."),
    "300": ("Murder", "101", "one_to_one", "yes", "Formal changes only (clauses (a)-(d))."),
    "302": ("Punishment for murder", "103(1)", "one_to_one", "yes", "Included as sub-section. New 103(2) (group murder on grounds of race, caste, etc.) has no IPC equivalent."),
    "304": ("Punishment for culpable homicide not amounting to murder", "105", "one_to_one", "yes", "Minimum 5 years added; fine mandatory."),
    "304A": ("Causing death by negligence", "106(1)", "one_to_one", "yes", "Imprisonment increased; registered medical practitioner clause added. New 106(2) (hit-and-run) is a separate new offence, kept on hold per the table."),
    "304B": ("Dowry death", "80", "one_to_one", "yes", "No change."),
    "306": ("Abetment of suicide", "108", "one_to_one", "yes", "No change."),
    "307": ("Attempt to murder", "109", "one_to_one", "yes", "Life-convict clause reworded (109(2))."),
    "308": ("Attempt to commit culpable homicide", "110", "one_to_one", "yes", "No change."),
    "313": ("Causing miscarriage without woman's consent", "89", "one_to_one", "yes", "Cross-reference wording only."),
    "320": ("Grievous hurt", "116", "modified", "yes", "Definition changed: 'twenty days' threshold reduced to fifteen days."),
    "323": ("Punishment for voluntarily causing hurt", "115(2)", "one_to_one", "yes", "Fine increased (up to Rs 10,000)."),
    "324": ("Voluntarily causing hurt by dangerous weapons or means", "118(1)", "one_to_one", "yes", "Fine increased (up to Rs 20,000)."),
    "325": ("Punishment for voluntarily causing grievous hurt", "117(2)", "one_to_one", "yes", "Formal changes. New 117(3)/(4) have no IPC equivalent."),
    "326": ("Voluntarily causing grievous hurt by dangerous weapons or means", "118(2)", "one_to_one", "yes", "Mandatory minimum 1 year added."),
    "332": ("Voluntarily causing hurt to deter public servant from his duty", "121(1)", "one_to_one", "yes", "Max imprisonment raised from 3 to 5 years."),
    "336": ("Act endangering life or personal safety of others", "125", "one_to_one", "yes", "Fine increased."),
    "337": ("Causing hurt by act endangering life or personal safety of others", "125(a)", "one_to_one", "yes", "Fine increased."),
    "338": ("Causing grievous hurt by act endangering life or personal safety of others", "125(b)", "one_to_one", "yes", "Imprisonment and fine increased."),
    "341": ("Punishment for wrongful restraint", "126(2)", "one_to_one", "yes", "Fine increased."),
    "342": ("Punishment for wrongful confinement", "127(2)", "one_to_one", "yes", "Fine increased."),
    "353": ("Assault or criminal force to deter public servant from discharge of his duty", "132", "one_to_one", "yes", "No change."),
    "354": ("Assault or criminal force to woman with intent to outrage her modesty", "74", "one_to_one", "yes", "No change."),
    "363": ("Punishment for kidnapping", "137(2)", "modified", "yes", "Punishment provision unchanged, but the kidnapping definition (IPC 361 -> BNS 137(1)(b)) now uses 'child' (gender-neutral, under 18), widening scope for boys aged 16-17."),
    "364": ("Kidnapping or abducting in order to murder", "140(1)", "one_to_one", "yes", "Included as sub-section."),
    "365": ("Kidnapping or abducting with intent secretly and wrongfully to confine person", "140(3)", "one_to_one", "yes", "'Ditto' in table."),
    "366": ("Kidnapping, abducting or inducing woman to compel her marriage, etc.", "87", "one_to_one", "yes", "No change. (CAPT subject line for BNS 87 is garbled in text extraction; IPC column clearly 366.)"),
    "366A": ("Procuration of minor girl", "96", "modified", "yes", "'Minor girl under 18' -> 'child': now gender-neutral."),
    "375": ("Rape", "63", "modified", "yes", "Definition changed: age in the marital exception raised from 15 to 18."),
    "376": ("Punishment for rape", "64", "one_to_one", "yes", "Table maps IPC 376(1) & 376(2) -> BNS 64. The sub-section label 376(2) has its mapping withheld."),
    "376(2)": ("Punishment for rape: sub-section (2), aggravated rape", "", "unclear", "no", "Mapping withheld: the table confirms IPC 376(1)&(2) -> BNS 64 only at section level; the sub-section-level correspondence could not be verified."),
    "379": ("Punishment for theft", "303(2)", "one_to_one", "yes", "Enhanced punishment for subsequent convictions added."),
    "380": ("Theft in dwelling house, etc.", "305", "modified", "yes", "Scope enlarged: means of transportation and place of worship added (new clauses (b)-(e))."),
    "384": ("Punishment for extortion", "308(2)", "one_to_one", "yes", "Max imprisonment raised from 3 to 7 years."),
    "389": ("Putting person in fear of accusation of offence, in order to commit extortion", "308(7)", "one_to_one", "yes", "'Ditto' in table."),
    "392": ("Punishment for robbery", "309(4)", "one_to_one", "yes", "No other change."),
    "394": ("Voluntarily causing hurt in committing robbery", "309(6)", "one_to_one", "yes", "'Ditto' in table."),
    "395": ("Punishment for dacoity", "310(2)", "one_to_one", "yes", "'Ditto' in table."),
    "397": ("Robbery or dacoity, with attempt to cause death or grievous hurt", "311", "one_to_one", "yes", "No change."),
    "406": ("Punishment for criminal breach of trust", "316(2)", "one_to_one", "yes", "Max imprisonment raised from 3 to 5 years."),
    "409": ("Criminal breach of trust by public servant, or by banker, merchant or agent", "316(5)", "one_to_one", "yes", "'Ditto' in table."),
    "411": ("Dishonestly receiving stolen property", "317(2)", "one_to_one", "yes", "Included as sub-section."),
    "415": ("Cheating", "318(1)", "one_to_one", "yes", "Definition presented as sub-section."),
    "417": ("Punishment for cheating", "318(2)", "one_to_one", "yes", "Max imprisonment raised from 1 to 3 years."),
    "419": ("Punishment for cheating by personation", "319(2)", "one_to_one", "yes", "Max imprisonment raised from 3 to 5 years."),
    "420": ("Cheating and dishonestly inducing delivery of property", "318(4)", "one_to_one", "yes", "Included as sub-section."),
    "427": ("Mischief causing damage to the amount of fifty rupees", "", "unclear", "no", "Mapping withheld: the BNS raised the damage thresholds (Rs 50 -> Rs 20,000 / 1 lakh), so the applicable sub-section (324(2)/(4)/(5)) depends on the amount; not verifiable from the table."),
    "436": ("Mischief by fire or explosive substance with intent to destroy house, etc.", "326(g)", "one_to_one", "yes", "Included as sub-section."),
    "437": ("Mischief with intent to destroy or make unsafe a decked vessel or one of twenty tons burden", "327(1)", "modified", "yes", "Scope widened: 'rail, aircraft' added."),
    "438": ("Punishment for the mischief described in section 437 committed by fire or explosive substance", "327(2)", "one_to_one", "yes", "Included as sub-section."),
    "447": ("Punishment for criminal trespass", "329(3)", "one_to_one", "yes", "Fine increased."),
    "448": ("Punishment for house-trespass", "329(4)", "one_to_one", "yes", "Fine increased."),
    "450": ("House-trespass in order to commit offence punishable with imprisonment for life", "332(b)", "one_to_one", "yes", "Included as sub-section."),
    "452": ("House-trespass after preparation for hurt, assault or wrongful restraint", "333", "one_to_one", "yes", "No change."),
    "457": ("Lurking house-trespass or house-breaking by night in order to commit offence punishable with imprisonment", "331(4)", "one_to_one", "yes", "'By night' -> 'after sunset and before sunrise'."),
    "465": ("Punishment for forgery", "336(2)", "one_to_one", "yes", "No other change."),
    "467": ("Forgery of valuable security, will, etc.", "338", "one_to_one", "yes", "No change."),
    "468": ("Forgery for purpose of cheating", "336(3)", "one_to_one", "yes", "'Ditto' in table."),
    "471": ("Using as genuine a forged document or electronic record", "340(2)", "one_to_one", "yes", "Included as sub-section."),
    "482": ("Punishment for using a false property mark", "345(3)", "one_to_one", "yes", "'Ditto' in table."),
    "494": ("Marrying again during lifetime of husband or wife", "82(1)", "one_to_one", "yes", "Included as sub-section."),
    "498": ("Enticing or taking away or detaining with criminal intent a married woman", "84", "modified", "yes", "Words 'from that man or from any person having the care of her on behalf of that man' removed."),
    "498A": ("Husband or relative of husband of a woman subjecting her to cruelty", "85", "one_to_one", "yes", "No change. (That 'cruelty' is defined in BNS 86 is from general knowledge, unverified here.)"),
    "500": ("Punishment for defamation", "356(2)", "one_to_one", "yes", "Community service added as alternative punishment."),
    "504": ("Intentional insult with intent to provoke breach of the peace", "352", "modified", "yes", "Words 'in any manner' added (scope)."),
    "506": ("Punishment for criminal intimidation", "351(2); 351(3)", "one_to_many", "yes", "Table: BNS 351(2)/(3) <- IPC 506 (the two limbs: simple vs threat of death/grievous hurt etc.)."),
    "509": ("Word, gesture or act intended to insult the modesty of a woman", "79", "one_to_one", "yes", "No change."),
    "511": ("Punishment for attempting to commit offences punishable with imprisonment for life or other imprisonment", "62", "one_to_one", "yes", "No change."),
}

# BNS provisions the CAPT table marks as new ('New' / 'Newly added' / no IPC section)
NEW_BNS = [
    ("48", "Abetment outside India for offence in India", "new provision"),
    ("69", "Sexual intercourse by employing deceitful means etc. (incl. false promise of marriage)", "new offence"),
    ("95", "Hiring, employing or engaging a child to commit an offence", "new offence"),
    ("103(2)", "Murder by a group of five or more on grounds of race, caste, community, sex, place of birth, language, personal belief ('mob lynching')", "new sub-clause"),
    ("106(2)", "Causing death by rash and negligent driving and escaping without reporting ('hit-and-run')", "new; table notes it was kept on hold, not in force from 1 July 2024"),
    ("111", "Organised crime", "new offence"),
    ("112", "Petty organised crime", "new offence"),
    ("113", "Terrorist act", "new offence"),
    ("117(3)", "Grievous hurt causing permanent disability or persistent vegetative state", "new sub-clause"),
    ("117(4)", "Grievous hurt by a group of five or more on grounds of race, caste, etc.", "new sub-clause"),
    ("152", "Act endangering sovereignty, unity and integrity of India", "described as newly added in the table (often discussed as replacing IPC 124A sedition; not in the ILSI label set)"),
    ("197(1)(d)", "Making or publishing false or misleading information jeopardising the sovereignty, unity, integrity or security of India", "new clause"),
    ("226", "Attempt to commit suicide to compel or restrain exercise of lawful power", "new offence"),
    ("304", "Snatching", "new offence"),
    ("324(3)", "Mischief causing loss or damage to any property including Government or local authority property", "new sub-section"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default=str(PROC))
    args = ap.parse_args()
    label_list = json.load(open(f"{args.proc_dir}/label_list.json"))
    missing = set(label_list) - set(ROWS)
    extra = set(ROWS) - set(label_list)
    assert not missing and not extra, f"mapping/label mismatch: missing={missing} extra={extra}"

    rows = []
    for ipc in label_list:
        title, bns, mtype, verified, notes = ROWS[ipc]
        source = ILSI if mtype == "repealed" else CAPT
        rows.append({"ipc_section": ipc, "ipc_title": title, "bns_sections": bns, "mapping_type": mtype,
                     "notes": notes, "source": source, "verified": verified})
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.proc_dir}/ipc_to_bns_mapping.csv", index=False)

    new = pd.DataFrame([{"bns_section": s, "title": t, "status": st, "ipc_equivalent": "none",
                         "training_data": "no training data (absent from ILSI label set)", "source": CAPT}
                        for s, t, st in NEW_BNS])
    new.to_csv(f"{args.proc_dir}/bns_new_offences.csv", index=False)

    print(df["mapping_type"].value_counts().to_string())
    print("verified:", df["verified"].value_counts().to_dict())
    print("UNVERIFIED rows:", df.loc[df.verified == "no", "ipc_section"].tolist())
    print(f"new BNS provisions listed: {len(new)}")


if __name__ == "__main__":
    main()
