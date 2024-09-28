
# Example sentences have a ton more content
# than just the idiom we're looking for. 
# In fact, they likely contain a lot of other idioms that our tags won't
# be able to catch. So, we need to trim the sentences down to just the idiom
# and some minor context. 
# ALTERNATIVELY, we could wait until we know all of the idioms, and then
# tag ALL idioms in every example sentence. This would take a lot more compute,
# but might squeeze more out of the training data.
class TrimPipeline:
    def process_item(self, item, spider):
        return item
