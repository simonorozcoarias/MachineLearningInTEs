import Bio
from Bio import SeqIO

def rebuild(tefile):
  f = open("repbase_joinedseq.fasta", "w")
  for te in SeqIO.parse(tefile, "fasta"):
    if not str(te.id).endswith("-I") and not str(te.id).endswith("_I") and not str(te.id).endswith("I"):
      print(te.id)
      LTRs = [ltrseq.seq for ltrseq in SeqIO.parse(tefile, "fasta") if str(ltrseq.id).find(str(te.id)[:-1]+"LTR") != -1]
      if len(LTRs) > 0:
        #print(LTRs[0])
        f.write(">"+str(te.id)[:-1]+"\n")
        f.write(str(LTRs[0])+str(te.seq)+str(LTRs[0])+"\n")


if __name__ == "__main__":
  rebuild("repbase_20170127.fasta")

