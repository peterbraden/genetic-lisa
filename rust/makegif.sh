convert -background black -loop 0 -delay 10 $(find rust/lisa -name "*.svg"  | sort -nr -t / -k 3 | xargs echo) lisa.gif && convert lisa.gif \( +clone -set delay 500 \) +swap +delete lisa.gif
